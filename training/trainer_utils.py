"""
Shared training utilities for all FakeNetBuster models.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file=None, level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )


class AverageMeter:
    """Tracks running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Stop training when a monitored metric stops improving.
    mode='min' for loss, mode='max' for AUC/accuracy."""
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.triggered = False

    def __call__(self, value):
        # Skip NaN/sentinel values
        if value is None or (isinstance(value, float) and (value != value or value >= 999.0)):
            return self.triggered
        if self.best is None:
            self.best = value
        elif self.mode == 'min':
            if value < self.best - self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # max
            if value > self.best + self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered


def save_checkpoint(model, optimizer, epoch, metric, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metric", 0.0)


def compute_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    labels = np.array(labels)
    classes = np.unique(labels)
    counts = np.bincount(labels.astype(int))
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
