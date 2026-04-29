"""
Deepfake Video Detection — EfficientNet-B3 frame backbone + BiLSTM temporal model.
Loads full dataset each run. Resumes from checkpoint.
Dataset layout:
  datasets/videos/train/{real,fake}/*.mp4
  datasets/videos/test/{real,fake}/*.mp4
"""

import os, sys, random, json, logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_video.model import VideoDeepfakeModel
from training.trainer_utils import (
    EarlyStopping, save_checkpoint, AverageMeter, setup_logging, set_seed
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path, n_frames=16, size=224):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size))
        frames.append(frame)
    cap.release()
    return frames


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoDataset(Dataset):
    def __init__(self, root_dir, split, n_frames=16, frame_size=224,
                 augment=True, max_per_class=None):
        self.n_frames   = n_frames
        self.frame_size = frame_size
        self.augment    = augment
        self.clips, self.labels = [], []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.aug_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ])

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = Path(root_dir) / split / cls
            if not cls_dir.exists():
                logger.warning(f"Missing: {cls_dir}")
                continue
            files = sorted(cls_dir.glob("*.mp4"))
            if max_per_class:
                files = files[:max_per_class]
            self.clips.extend([(f, label) for f in files])
            self.labels.extend([label] * len(files))

        r = self.labels.count(0)
        f = self.labels.count(1)
        logger.info(f"[{split}] {len(self.clips)} clips — real: {r}, fake: {f}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        path, label = self.clips[idx]
        frames = extract_frames(path, self.n_frames, self.frame_size)
        if frames is None:
            return (torch.zeros(self.n_frames, 3, self.frame_size, self.frame_size),
                    torch.tensor(label, dtype=torch.float32))
        tensors = []
        for frame in frames:
            if self.augment:
                if random.random() > 0.5:
                    frame = cv2.flip(frame, 1)
                if random.random() > 0.6:
                    # color jitter via numpy
                    frame = np.clip(frame.astype(np.float32) * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
                tensors.append(self.aug_transform(frame))
            else:
                tensors.append(self.transform(frame))
        return torch.stack(tensors), torch.tensor(label, dtype=torch.float32)


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for clips, labels in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=scaler.is_enabled()):
            logits, _ = model(clips)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), clips.size(0))
        preds = (torch.sigmoid(logits).detach().cpu().float() > 0.5).numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())
        pbar.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg, accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
        for clips, labels in pbar:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(clips)
            loss = criterion(logits.float(), labels.float())
            meter.update(loss.item(), clips.size(0))
            p = torch.sigmoid(logits).cpu().float().numpy().flatten()
            all_probs.extend(p)
            all_preds.extend((p > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy().flatten())
            pbar.set_postfix(loss=f"{meter.avg:.4f}")
    acc = accuracy_score(all_labels, all_preds)
    all_probs = np.nan_to_num(all_probs, nan=0.5)  # guard against NaN
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    val_loss = meter.avg if not np.isnan(meter.avg) else 999.0  # NaN loss -> large value so early stop ignores it
    return val_loss, acc, auc


# ── Main ──────────────────────────────────────────────────────────────────────

def train(config_path="configs/model_configs.yaml",
          training_config_path="configs/training_configs.yaml",
          data_root="datasets/videos",
          save_dir="saved_models",
          freeze_epochs=2):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["video_model"]
    with open(training_config_path) as f:
        tcfg = yaml.safe_load(f)

    set_seed(tcfg["general"]["seed"])
    setup_logging("logs/video_training.log")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = GradScaler('cuda', enabled=use_amp)
    logger.info(f"Device: {device}")

    n_frames   = cfg.get("sequence_length", 16)
    frame_size = cfg.get("frame_size", 224)
    batch_size = cfg.get("batch_size", 2)
    epochs     = cfg.get("epochs", 20)
    lr         = cfg.get("learning_rate", 1e-4)

    train_ds = VideoDataset(data_root, "train", n_frames, frame_size, augment=True)
    test_ds  = VideoDataset(data_root, "test",  n_frames, frame_size, augment=False)

    if len(train_ds) == 0:
        logger.error("No training videos found.")
        return

    # Balanced sampler
    weights = [1.0 / max(train_ds.labels.count(l), 1) for l in train_ds.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   num_workers=0)

    model = VideoDeepfakeModel(
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
    ).to(device)

    # No backbone freeze — train everything from epoch 1 with low LR
    freeze_epochs = 0
    logger.info("Training all layers from epoch 1 (no backbone freeze)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=tcfg["general"]["early_stopping_patience"], mode='max')

    # Resume
    ckpt_path = os.path.join(save_dir, "video_model_best.pth")
    best_auc, start_epoch = 0.0, 1
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_auc    = ckpt.get("metric", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed epoch {start_epoch}, best AUC: {best_auc:.4f}")

    end_epoch = start_epoch + epochs
    for epoch in tqdm(range(start_epoch, end_epoch),
                      desc=f"Epochs ({start_epoch}->{end_epoch-1})", unit="epoch"):

        if epoch == start_epoch + freeze_epochs and freeze_epochs > 0:
            model.freeze_backbone(False)
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": lr * 0.01},
                {"params": list(model.lstm.parameters()) +
                           list(model.attn.parameters()) +
                           list(model.head.parameters()), "lr": lr * 0.5},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs)
            logger.info("Backbone unfrozen")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc, val_auc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f}")

        save_checkpoint(model, optimizer, epoch, val_auc,
                        os.path.join(save_dir, "video_model_latest.pth"))
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc, ckpt_path)
            logger.info(f"  New best AUC={best_auc:.4f} saved")

        if early_stop(val_auc):  # monitor AUC, not loss
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Done. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
