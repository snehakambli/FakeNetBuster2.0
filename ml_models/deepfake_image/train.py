"""
Training pipeline for Deepfake Image Detection — EfficientNet-B4 + Frequency + Noise.
Dataset layout:
  datasets/images/train/{real,fake}/
  datasets/images/validation/{real,fake}/
  datasets/images/test/{real,fake}/

Features:
  - Mixed-precision (AMP) training
  - Backbone freeze for first N epochs, then full fine-tune
  - Weighted sampler for class imbalance
  - Label smoothing + BCEWithLogitsLoss with pos_weight
  - Cosine LR schedule with warm-up
  - Early stopping on validation AUC
  - Checkpoint resume
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_image.model import build_model
from training.trainer_utils import (
    EarlyStopping, save_checkpoint, load_checkpoint,
    AverageMeter, setup_logging, set_seed
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset ──────────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    """
    Loads all images from root_dir/split/{real,fake}/.
    label: 0 = real, 1 = fake
    """
    def __init__(self, root_dir, split="train", transform=None, max_per_class=None):
        self.transform = transform
        self.samples, self.labels = [], []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = Path(root_dir) / split / cls
            if not cls_dir.exists():
                logger.warning(f"Missing: {cls_dir}")
                continue
            files = sorted([
                str(p) for p in cls_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ])
            if max_per_class:
                files = files[:max_per_class]
            self.samples.extend(files)
            self.labels.extend([label] * len(files))

        real_n = self.labels.count(0)
        fake_n = self.labels.count(1)
        logger.info(f"[{split}] {len(self.samples)} images — real: {real_n}, fake: {fake_n}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (380, 380))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(input_size=380, augment=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if augment:
        return transforms.Compose([
            transforms.Resize((input_size + 40, input_size + 40)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Warmup + Cosine LR ───────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / Eval loops ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler,
                mid_epoch_ckpt_path=None, save_every=500):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast('cuda', enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss   = criterion(logits, labels)

                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.error(f"CUDA OOM at batch {batch_idx} — skipping batch. Reduce batch_size or input_size.")
                continue
            raise

        meter.update(loss.item(), imgs.size(0))
        probs = torch.sigmoid(logits).detach().float().cpu().numpy().flatten()
        all_preds.extend((probs > 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy().flatten().astype(int))
        pbar.set_postfix(loss=f"{meter.avg:.4f}")

        # Mid-epoch checkpoint every N batches
        if mid_epoch_ckpt_path and (batch_idx + 1) % save_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_idx": batch_idx,
                "loss": meter.avg,
            }, mid_epoch_ckpt_path)

    acc = accuracy_score(all_labels, all_preds)
    return meter.avg, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(imgs)
            loss = criterion(logits.float(), labels.float())
            meter.update(loss.item(), imgs.size(0))
            probs = torch.sigmoid(logits).float().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
            pbar.set_postfix(loss=f"{meter.avg:.4f}")

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return meter.avg, acc, auc, f1


# ── Main train function ───────────────────────────────────────────────────────

def train(
    config_path="configs/model_configs.yaml",
    training_config_path="configs/training_configs.yaml",
    data_root="datasets/images",
    save_dir="saved_models",
    freeze_backbone_epochs=3,
    max_per_class=None,   # None = use all images
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["image_model"]
    with open(training_config_path) as f:
        tcfg = yaml.safe_load(f)

    set_seed(tcfg["general"]["seed"])
    setup_logging("logs/image_training.log")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and tcfg["general"]["device"] == "cuda" else "cpu"
    )
    logger.info(f"Device: {device}")
    use_amp = device.type == "cuda"
    scaler  = GradScaler('cuda', enabled=use_amp)

    input_size = cfg.get("input_size", 380)
    batch_size = cfg.get("batch_size", 32)
    epochs     = cfg.get("epochs", 20)
    lr         = cfg.get("learning_rate", 3e-4)
    logger.info(f"batch_size={batch_size}, input_size={input_size}")

    # ── Datasets ──
    train_ds = DeepfakeImageDataset(data_root, "train",
                                    get_transforms(input_size, augment=True),
                                    max_per_class=max_per_class)
    val_ds   = DeepfakeImageDataset(data_root, "validation",
                                    get_transforms(input_size, augment=False),
                                    max_per_class=2000)
    test_ds  = DeepfakeImageDataset(data_root, "test",
                                    get_transforms(input_size, augment=False),
                                    max_per_class=2000)

    # Data is already balanced 50/50 — plain shuffle, no weighted sampler.
    # WeightedRandomSampler + pos_weight together caused 100% fake predictions.
    num_workers = 0  # Windows: must be 0 to avoid multiprocessing spawn hang
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_amp)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_amp)

    # ── Model ──
    model = build_model(cfg).to(device)

    # Freeze backbone for first N epochs (faster warm-up)
    if freeze_backbone_epochs > 0:
        model.freeze_backbone(True)
        logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs")

    # No pos_weight — WeightedRandomSampler already balances classes.
    # Adding pos_weight on top causes double-weighting that biases all
    # predictions toward "fake" (high AUC but 100% fake predictions).
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=tcfg["optimizer"].get("weight_decay", 1e-4)
    )
    scheduler   = get_scheduler(optimizer, warmup_epochs=tcfg["optimizer"].get("warmup_epochs", 3), total_epochs=epochs)
    early_stop  = EarlyStopping(patience=tcfg["general"]["early_stopping_patience"], mode='max')

    # ── Fresh start — delete broken checkpoint if exists ──
    checkpoint_path = os.path.join(save_dir, "image_model_best.pth")
    best_auc, start_epoch = 0.0, 1
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Removed old checkpoint — starting fresh")

    # ── Training loop ──
    end_epoch = start_epoch + epochs
    history = []
    for epoch in tqdm(range(start_epoch, end_epoch), desc=f"Epochs ({start_epoch}→{end_epoch-1})", unit="epoch"):

        # Unfreeze backbone after freeze_backbone_epochs
        if epoch == start_epoch + freeze_backbone_epochs:
            model.freeze_backbone(False)
            # Enable gradient checkpointing to save VRAM during full fine-tune
            try:
                model.spatial_features.gradient_checkpointing = True
                for module in model.spatial_features.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
            except Exception:
                pass
            torch.cuda.empty_cache()
            # Re-init optimizer with all params + lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.spatial_features.parameters(), "lr": lr * 0.05},
                {"params": list(model.freq_branch.parameters()) +
                           list(model.noise_branch.parameters()) +
                           list(model.classifier.parameters()), "lr": lr},
            ], weight_decay=tcfg["optimizer"].get("weight_decay", 1e-4))
            scheduler = get_scheduler(optimizer, warmup_epochs=1, total_epochs=epochs - freeze_backbone_epochs)
            logger.info("Backbone unfrozen — fine-tuning all layers")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                            mid_epoch_ckpt_path=None,
                                            save_every=9999)
        val_loss, val_acc, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(train_loss,4), train_acc=round(train_acc,4),
                   val_loss=round(val_loss,4), val_acc=round(val_acc,4),
                   val_auc=round(val_auc,4), val_f1=round(val_f1,4))
        history.append(row)
        logger.info(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f} F1 {val_f1:.4f}")

        # Collapse detection — if val_acc is near 50% after epoch 3, something is wrong
        if epoch >= 3 and val_acc < 0.55:
            logger.warning(f"WARNING: val_acc={val_acc:.3f} — possible prediction collapse. Check model.")

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc,
                            os.path.join(save_dir, "image_model_best.pth"))
            logger.info(f"  New best AUC={best_auc:.4f} saved")

        if early_stop(val_auc):
            logger.info("Early stopping triggered.")
            break

    # Save training history
    with open("logs/image_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final test evaluation ──
    logger.info("Running final evaluation on test set...")
    test_loss, test_acc, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test | Loss {test_loss:.4f} Acc {test_acc:.4f} AUC {test_auc:.4f} F1 {test_f1:.4f}")

    logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")
    return model


if __name__ == "__main__":
    # Fast training: 5k per class (10k total), 15 epochs, no resume
    # Fixes: no weighted sampler, neutral loss — no prediction collapse
    train(max_per_class=5000, freeze_backbone_epochs=2)
