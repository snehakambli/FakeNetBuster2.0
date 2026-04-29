"""
Deepfake Audio Detection — CNN spectrogram + BiGRU temporal model.
Dataset: datasets/audio/train/{real,fake}/*.wav
"""

import os, sys, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_audio.model import build_model
from training.trainer_utils import (
    EarlyStopping, save_checkpoint, AverageMeter, setup_logging, set_seed
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    def __init__(self, root_dir, split="train", sample_rate=16000,
                 n_mels=128, n_fft=1024, hop_length=512,
                 max_frames=128, augment=False):
        self.sr, self.n_mels = sample_rate, n_mels
        self.n_fft, self.hop = n_fft, hop_length
        self.max_frames = max_frames
        self.augment = augment
        self.samples, self.labels = [], []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = Path(root_dir) / split / cls
            if not cls_dir.exists():
                logger.warning(f"Missing: {cls_dir}")
                continue
            for f in sorted(f for ext in ("*.wav", "*.flac") for f in cls_dir.glob(ext)):
                self.samples.append(str(f))
                self.labels.append(label)

        r, f = self.labels.count(0), self.labels.count(1)
        logger.info(f"[{split}] {len(self.samples)} files  real={r}  fake={f}")

    def _load_mel(self, path):
        try:
            y, sr = librosa.load(path, sr=self.sr, mono=True)
        except Exception:
            y = np.zeros(self.sr, dtype=np.float32)
            sr = self.sr

        if self.augment:
            if np.random.random() > 0.5:
                y = y + (np.random.randn(len(y)) * 0.005).astype(np.float32)
            if np.random.random() > 0.5:
                rate = np.random.uniform(0.90, 1.10)
                y = librosa.effects.time_stretch(y, rate=rate)
            if np.random.random() > 0.7:
                # random time masking
                mask_len = int(len(y) * np.random.uniform(0.05, 0.15))
                start = np.random.randint(0, max(1, len(y) - mask_len))
                y[start:start + mask_len] = 0.0
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        # SpecAugment: frequency and time masking
        if self.augment:
            # Frequency masking — zero out up to 20 mel bins
            f_mask = np.random.randint(0, 20)
            f_start = np.random.randint(0, max(1, self.n_mels - f_mask))
            mel_db[f_start:f_start + f_mask, :] = 0.0
            # Time masking — zero out up to 30 time frames
            t_mask = np.random.randint(0, 30)
            t_start = np.random.randint(0, max(1, mel_db.shape[1] - t_mask))
            mel_db[:, t_start:t_start + t_mask] = 0.0

        if mel_db.shape[1] < self.max_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_frames - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :self.max_frames]

        return mel_db.astype(np.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        spec = self._load_mel(self.samples[idx])
        return torch.tensor(spec).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []
    for specs, labels in tqdm(loader, desc="  Train", leave=False):
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=scaler.is_enabled()):
            logits, _ = model(specs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if scheduler is not None:
            scheduler.step()  # OneCycleLR steps per batch
        meter.update(loss.item(), specs.size(0))
        preds = (torch.sigmoid(logits).detach().cpu().float() > 0.5).numpy().flatten()
        all_preds.extend(preds); all_labels.extend(labels.cpu().numpy().flatten())
    return meter.avg, accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="  Val  ", leave=False):
            specs  = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast('cuda', enabled=(device.type == "cuda")):
                logits, _ = model(specs)
            loss = criterion(logits.float(), labels.float())
            meter.update(loss.item(), specs.size(0))
            p = torch.sigmoid(logits).cpu().float().numpy().flatten()
            all_probs.extend(p); all_preds.extend((p > 0.5).astype(int))
            all_labels.extend(labels.cpu().numpy().flatten())
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, np.nan_to_num(all_probs, nan=0.5)) if len(set(all_labels)) > 1 else 0.0
    return meter.avg if not np.isnan(meter.avg) else 999.0, acc, auc


def train(config_path="configs/model_configs.yaml",
          training_config_path="configs/training_configs.yaml",
          data_root="datasets/audio", save_dir="saved_models"):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["audio_model"]
    with open(training_config_path) as f:
        tcfg = yaml.safe_load(f)

    set_seed(tcfg["general"]["seed"])
    setup_logging("logs/audio_training.log")
    os.makedirs(save_dir, exist_ok=True); os.makedirs("logs", exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = GradScaler('cuda', enabled=use_amp)
    logger.info(f"Device: {device}")

    # max_frames must match predict.py MAX_FRAMES = 128
    max_frames = cfg.get("max_frames", 128)

    train_ds = AudioDataset(data_root, "train", cfg["sample_rate"], cfg["n_mels"],
                            cfg["n_fft"], cfg["hop_length"], max_frames, augment=True)
    test_ds  = AudioDataset(data_root, "test",  cfg["sample_rate"], cfg["n_mels"],
                            cfg["n_fft"], cfg["hop_length"], max_frames, augment=False)

    if len(train_ds) == 0:
        logger.error("No training audio found. Run datasets/arrange_audio_dataset.py first.")
        return

    # Data is balanced — plain shuffle, no weighted sampler (causes bias toward one class)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model     = build_model(cfg).to(device)
    # No pos_weight — WeightedRandomSampler already balances classes.
    # Adding pos_weight on top causes double-weighting that biases all
    # predictions toward "fake" (high AUC but 100% fake predictions).
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_epochs = cfg.get("epochs", 30)
    # Warmup 10% then cosine decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, epochs=total_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    early_stop = EarlyStopping(patience=10, mode='max')

    ckpt_path = os.path.join(save_dir, "audio_model_best.pth")
    best_auc, start_epoch = 0.0, 1

    # Fresh start — remove broken checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("Removed old checkpoint — starting fresh")

    for epoch in tqdm(range(start_epoch, start_epoch + total_epochs), desc="Epochs", unit="epoch"):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device, scheduler)
        val_loss, val_acc, val_auc = evaluate(model, test_loader, criterion, device)

        logger.info(f"Epoch {epoch} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f}")

        save_checkpoint(model, optimizer, epoch, val_auc,
                        os.path.join(save_dir, "audio_model_latest.pth"))
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc, ckpt_path)
            logger.info(f"  New best AUC={best_auc:.4f} saved")

        if early_stop(val_auc):
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Done. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
