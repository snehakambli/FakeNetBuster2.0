"""
Training pipeline for Fake Document Detection.
Dataset: MIDV dataset / RVL-CDIP
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.fake_documents.model import build_model, DocumentTokenizer
from training.trainer_utils import (
    EarlyStopping, save_checkpoint, AverageMeter, setup_logging, set_seed
)

logger = logging.getLogger(__name__)


class DocumentDataset(Dataset):
    """
    Loads document images from:
    root/train/real/*.jpg
    root/train/fake/*.jpg
    Optionally loads OCR text from companion .txt files.
    """
    def __init__(self, root_dir, split="train", input_size=256,
                 tokenizer=None, max_text_len=256, augment=True):
        self.input_size = input_size
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.samples = []
        self.labels = []

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((input_size + 40, input_size + 40)),
                transforms.RandomCrop(input_size),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
                transforms.RandomGrayscale(p=0.08),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.15, scale=(0.01, 0.06)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

        logger.info(f"[{split}] Loaded {len(self.samples)} document images")

    def _load_ocr_text(self, img_path):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img_tensor = self.transform(img)

        token_ids = None
        if self.tokenizer:
            text = self._load_ocr_text(self.samples[idx])
            ids = self.tokenizer.encode(text, self.max_text_len)
            token_ids = torch.tensor(ids, dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img_tensor, token_ids, label


def collate_fn(batch):
    imgs, tokens, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    # Only stack tokens if ALL items have them
    if all(t is not None for t in tokens):
        tokens = torch.stack(tokens)
    else:
        tokens = None
    return imgs, tokens, labels


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []

    for imgs, tokens, labels in tqdm(loader, desc="  Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        if tokens is not None:
            tokens = tokens.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device.type == "cuda")):
            probs = model(imgs, tokens)
            loss = criterion(probs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        meter.update(loss.item(), imgs.size(0))
        preds = (torch.sigmoid(probs).detach().cpu().float() > 0.5).numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    return meter.avg, accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, tokens, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            if tokens is not None:
                tokens = tokens.to(device, non_blocking=True)

            with autocast('cuda', enabled=(device.type == "cuda")):
                probs = model(imgs, tokens)
            loss = criterion(probs.float(), labels.float())

            meter.update(loss.item(), imgs.size(0))
            p = torch.sigmoid(probs).cpu().float().numpy().flatten()
            all_probs.extend(p)
            all_preds.extend((p > 0.5).astype(float))
            all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return meter.avg, acc, auc


def train(config_path="configs/model_configs.yaml",
          training_config_path="configs/training_configs.yaml",
          data_root="datasets/documents",
          save_dir="saved_models"):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["document_model"]
    with open(training_config_path) as f:
        tcfg = yaml.safe_load(f)

    set_seed(tcfg["general"]["seed"])
    setup_logging("logs/document_training.log")
    os.makedirs(save_dir, exist_ok=True)

    # cuDNN: deterministic mode for reproducibility
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = DocumentTokenizer()
    tokenizer.save(os.path.join(save_dir, "document_tokenizer.json"))

    train_ds = DocumentDataset(data_root, "train", cfg["input_size"],
                               tokenizer, augment=True)
    val_ds   = DocumentDataset(data_root, "validation", cfg["input_size"],
                               tokenizer, augment=False)
    test_ds  = DocumentDataset(data_root, "test", cfg["input_size"],
                               tokenizer, augment=False)

    from torch.utils.data import WeightedRandomSampler
    real_n = train_ds.labels.count(0)
    fake_n = train_ds.labels.count(1)
    sample_weights = [1.0 / real_n if l == 0 else 1.0 / fake_n for l in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # num_workers=0: avoids Windows multiprocessing spawn hang
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=0,
                              pin_memory=False,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0,
                              pin_memory=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0,
                              pin_memory=False,
                              collate_fn=collate_fn)

    model     = build_model(cfg).to(device)
    pos_weight = torch.tensor([real_n / max(fake_n, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler    = GradScaler('cuda', enabled=(device.type == "cuda"))  # AMP gradient scaler
    early_stop = EarlyStopping(patience=tcfg["general"]["early_stopping_patience"], mode='max')

    best_auc = 0.0
    for epoch in tqdm(range(1, cfg["epochs"] + 1), desc="Epochs", unit="epoch"):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, scaler, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{cfg['epochs']} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc,
                            os.path.join(save_dir, "document_model_best.pth"))

        if early_stop(val_auc):
            logger.info("Early stopping triggered.")
            break

        # free unused VRAM between epochs
        torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, epoch, val_auc,
                    os.path.join(save_dir, "document_model_final.pth"))

    # Final test evaluation
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test | Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")
    logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
