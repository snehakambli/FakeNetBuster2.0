"""
Deepfake Video Detection Model
Architecture: EfficientNet-B3 frame backbone + BiLSTM + attention
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class VideoDeepfakeModel(nn.Module):
    """
    EfficientNet-B3 per-frame features (1536-d) + BiLSTM temporal + attention.
    Input : (B, T, 3, 224, 224)
    Output: (B, 1) raw logit, (B, T) attention weights
    """
    def __init__(self, lstm_hidden=256, lstm_layers=2, dropout=0.4):
        super().__init__()
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool     = backbone.avgpool
        self.feat_dim = 1536  # EfficientNet-B3 output channels

        self.lstm = nn.LSTM(
            self.feat_dim, lstm_hidden, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        feat = self.features(frames)
        feat = self.pool(feat).view(B * T, self.feat_dim)
        feat = feat.view(B, T, self.feat_dim)

        lstm_out, _ = self.lstm(feat)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        ctx    = (lstm_out * attn_w).sum(dim=1)
        return self.head(ctx), attn_w.squeeze(-1)

    def freeze_backbone(self, freeze=True):
        for p in self.features.parameters():
            p.requires_grad = not freeze


def build_model(config=None):
    hidden = config.get("lstm_hidden", 256) if config else 256
    layers = config.get("lstm_layers", 2)   if config else 2
    return VideoDeepfakeModel(lstm_hidden=hidden, lstm_layers=layers)
