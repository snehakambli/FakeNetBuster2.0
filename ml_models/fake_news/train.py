"""
Training pipeline for Fake News Detection.
Dataset: LIAR dataset / FakeNewsNet / multilingual fake news datasets
"""

import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import yaml
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.fake_news.model import build_model, SimpleTokenizer
from training.trainer_utils import (
    EarlyStopping, save_checkpoint, AverageMeter, setup_logging, set_seed
)

logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """
    Loads news text from CSV/JSON files.
    Expected format: {"text": "...", "label": 0/1}
    Label: 0=real, 1=fake
    """
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.labels = []
        self._load(data_path)

    def _load(self, path):
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}")
            return

        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            for item in data:
                text = item.get("text", item.get("statement", ""))
                label = int(item.get("label", 0))
                if text:
                    self.samples.append(text)
                    self.labels.append(label)

        elif path.endswith(".csv") or path.endswith(".tsv"):
            sep = "\t" if path.endswith(".tsv") else ","
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    text = row.get("text", row.get("statement", row.get("title", "")))
                    label_str = row.get("label", "0")
                    # LIAR dataset: pants-fire/false/barely-true -> fake, true/mostly-true -> real
                    if label_str in ["pants-fire", "false", "barely-true", "1", "fake"]:
                        label = 1
                    else:
                        label = 0
                    if text:
                        self.samples.append(text)
                        self.labels.append(label)

        logger.info(f"Loaded {len(self.samples)} samples "
                    f"({self.labels.count(0)} real, {self.labels.count(1)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask = self.tokenizer.encode(self.samples[idx], self.max_len)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []

    for ids, mask, labels in loader:
        ids, mask = ids.to(device), mask.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        probs, _ = model(ids, mask)
        loss = criterion(probs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        meter.update(loss.item(), ids.size(0))
        preds = (probs.detach().cpu() > 0.5).float().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    return meter.avg, accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask = ids.to(device), mask.to(device)
            labels = labels.to(device).unsqueeze(1)
            probs, _ = model(ids, mask)
            loss = criterion(probs, labels)
            meter.update(loss.item(), ids.size(0))

            p = probs.cpu().numpy().flatten()
            all_probs.extend(p)
            all_preds.extend((p > 0.5).astype(float))
            all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return meter.avg, acc, auc, f1


def train(config_path="configs/model_configs.yaml",
          training_config_path="configs/training_configs.yaml",
          data_root="datasets/news",
          save_dir="saved_models"):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["news_model"]
    with open(training_config_path) as f:
        tcfg = yaml.safe_load(f)

    set_seed(tcfg["general"]["seed"])
    setup_logging("logs/news_training.log")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Build tokenizer from training data
    tokenizer = SimpleTokenizer(vocab_size=cfg["vocab_size"], max_len=cfg["max_length"])
    train_file = os.path.join(data_root, "train.json")
    test_file = os.path.join(data_root, "test.json")

    # Fit tokenizer on training texts
    if os.path.exists(train_file):
        with open(train_file) as f:
            train_data = json.load(f)
        texts = [item.get("text", item.get("statement", "")) for item in train_data]
        tokenizer.fit(texts)
    else:
        logger.warning("Training file not found, using empty tokenizer")
        tokenizer.fitted = True

    tokenizer.save(os.path.join(save_dir, "news_tokenizer.json"))

    train_ds = NewsDataset(train_file, tokenizer, cfg["max_length"])
    test_ds = NewsDataset(test_file, tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=4)

    model = build_model(cfg).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    early_stop = EarlyStopping(patience=tcfg["general"]["early_stopping_patience"])

    best_auc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_f1 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{cfg['epochs']} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"AUC: {val_auc:.4f} F1: {val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc,
                            os.path.join(save_dir, "news_model_best.pth"))

        if early_stop(val_loss):
            logger.info("Early stopping triggered.")
            break

    save_checkpoint(model, optimizer, epoch, val_auc,
                    os.path.join(save_dir, "news_model_final.pth"))
    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
