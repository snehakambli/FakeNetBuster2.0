"""
Deepfake Image Detection Model — EfficientNet-B4 backbone + Frequency + Noise branches.
Uses torchvision's EfficientNet_B4 (pretrained on ImageNet) as the spatial backbone.
Triple-branch fusion: Spatial(1792) + Frequency(128) + Noise(64) → logit
Output: raw logit (apply sigmoid at inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class FrequencyBranch(nn.Module):
    """High-pass + Laplacian frequency analysis for GAN fingerprints."""
    def __init__(self):
        super().__init__()
        hp  = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('hp_kernel', hp)
        self.register_buffer('lap_kernel', lap)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        hp   = F.conv2d(gray, self.hp_kernel, padding=1)
        lap  = F.conv2d(gray, self.lap_kernel, padding=1)
        out  = self.net(torch.cat([hp, lap], dim=1))
        return F.relu(self.fc(out.view(out.size(0), -1)))   # (B, 128)


class NoiseBranch(nn.Module):
    """SRM noise residual branch for splicing/generation artifacts."""
    def __init__(self):
        super().__init__()
        srm = torch.zeros(3, 3, 5, 5)
        k   = torch.tensor([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                             [0,-1,2,-1,0],[0,0,0,0,0]], dtype=torch.float32)
        for i in range(3):
            srm[i, i] = k
        self.register_buffer('srm', srm)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        noise = F.conv2d(x, self.srm, padding=2)
        out   = self.net(noise)
        return F.relu(self.fc(out.view(out.size(0), -1)))   # (B, 64)


class DeepfakeImageModel(nn.Module):
    """
    EfficientNet-B4 backbone + Frequency(128) + Noise(64) branches.
    Spatial features: 1792-dim from EfficientNet-B4 global pool.
    Total fusion: 1792 + 128 + 64 = 1984 → 512 → 128 → 1 (logit)
    Input:  (B, 3, 380, 380)  [EfficientNet-B4 native size]
    Output: (B, 1) raw logit
    """
    def __init__(self, dropout=0.4, freeze_backbone_epochs=0):
        super().__init__()
        # EfficientNet-B4 backbone 
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # Remove the classifier head, keep features + avgpool
        self.spatial_features = backbone.features   # outputs (B, 1792, H, W)
        self.spatial_pool     = backbone.avgpool     # AdaptiveAvgPool2d(1)
        self.spatial_dim      = 1792

        self.freq_branch  = FrequencyBranch()
        self.noise_branch = NoiseBranch()

        fusion_dim = self.spatial_dim + 128 + 64  # 1984
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128),        nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
        # Attention gate to weight branch contributions
        self.branch_attn = nn.Sequential(
            nn.Linear(fusion_dim, 3), nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Spatial branch
        s = self.spatial_features(x)
        s = self.spatial_pool(s).view(s.size(0), -1)   # (B, 1792)
        # Frequency & noise branches
        f = self.freq_branch(x)    # (B, 128)
        n = self.noise_branch(x)   # (B, 64)
        fused = torch.cat([s, f, n], dim=1)  # (B, 1984)
        return self.classifier(fused)        # (B, 1)

    def get_feature_maps(self, x):
        """Return last spatial feature map for GradCAM."""
        return self.spatial_features(x)

    def freeze_backbone(self, freeze=True):
        for p in self.spatial_features.parameters():
            p.requires_grad = not freeze


def build_model(config=None):
    dropout = config.get("dropout", 0.4) if config else 0.4
    return DeepfakeImageModel(dropout=dropout)
