"""
Deepfake Audio Detection Model
Architecture: CNN (spectrogram features) + BiGRU temporal modeling
"""

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, None)),
        )
        self.freq_compress = nn.Linear(256 * 4, 256)

    def forward(self, x):
        x = self.features(x)              # (B, 256, 4, T)
        B, C, nF, T = x.shape
        x = x.permute(0, 3, 1, 2)        # (B, T, C, nF)
        x = x.reshape(B, T, C * nF)      # (B, T, 1024)
        x = torch.relu(self.freq_compress(x))  # (B, T, 256)
        return x


class DeepfakeAudioModel(nn.Module):
    def __init__(self, n_mels=128, gru_hidden=256, gru_layers=2, dropout=0.4):
        super().__init__()
        self.cnn = SpectrogramCNN(n_mels=n_mels)
        self.gru = nn.GRU(256, gru_hidden, gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.attn = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1)
            # No Sigmoid — BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x):
        feat = self.cnn(x)                           # (B, T, 256)
        gru_out, _ = self.gru(feat)                  # (B, T, hidden*2)
        attn_w = torch.softmax(self.attn(gru_out), dim=1)
        ctx = (gru_out * attn_w).sum(dim=1)
        return self.head(ctx), attn_w.squeeze(-1)


def build_model(config=None):
    n_mels     = config.get("n_mels", 128)     if config else 128
    gru_hidden = config.get("gru_hidden", 256) if config else 256
    gru_layers = config.get("gru_layers", 2)   if config else 2
    return DeepfakeAudioModel(n_mels=n_mels, gru_hidden=gru_hidden, gru_layers=gru_layers, dropout=0.5)
