"""
Fake Document Detection Model
Architecture: CNN (visual forgery) + NLP classifier (text authenticity)
Detects: tampered regions, font inconsistencies, fake seals, layout anomalies
Supports: Aadhar cards, certificates, ID cards (MIDV / RVL-CDIP datasets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentCNN(nn.Module):
    """
    CNN for visual document forgery detection.
    Analyzes layout, seal regions, font consistency, and pixel tampering.
    Input: (B, 3, 512, 512)
    Output: (B, 512) visual features
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 - detect low-level pixel artifacts
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x256

            # Block 2 - detect font and text region inconsistencies
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128

            # Block 3 - detect seal and stamp regions
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64

            # Block 4 - high-level layout analysis
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32

            # Block 5
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_feature_maps(self, x):
        return self.features(x)


class TextAuthenticityClassifier(nn.Module):
    """
    NLP classifier for OCR-extracted text authenticity.
    Detects: inconsistent formatting, suspicious patterns, invalid data.
    Input: (B, text_feat_dim) pre-extracted text features
    Output: (B, 256) text features
    """
    def __init__(self, vocab_size=10000, embed_dim=128, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, 256)

    def forward(self, token_ids):
        x = self.embedding(token_ids)       # (B, L, embed_dim)
        x = x.transpose(1, 2)              # (B, embed_dim, L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)       # (B, 256)
        return F.relu(self.fc(x))


class FakeDocumentModel(nn.Module):
    """
    Dual-branch document forgery detection model.
    Combines visual CNN analysis with text authenticity classification.
    """
    def __init__(self, vocab_size=10000, embed_dim=128, max_len=256):
        super().__init__()
        self.visual_branch = DocumentCNN()
        self.text_branch = TextAuthenticityClassifier(vocab_size, embed_dim, max_len)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Visual-only head (when no OCR available)
        self.visual_only_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image, token_ids=None):
        visual_feat = self.visual_branch(image)

        if token_ids is not None:
            text_feat = self.text_branch(token_ids)
            combined = torch.cat([visual_feat, text_feat], dim=1)
            return self.fusion(combined)
        else:
            return self.visual_only_head(visual_feat)

    def get_visual_features(self, image):
        return self.visual_branch.get_feature_maps(image)


class DocumentTokenizer:
    """Simple tokenizer for OCR-extracted document text."""
    def __init__(self, vocab_size=10000, max_len=256):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        # Character-level tokenization for robustness with OCR noise
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 .,/-:"):
            self.char2idx[c] = i + 2

    def encode(self, text, max_len=None):
        max_len = max_len or self.max_len
        text = text.lower()[:max_len]
        ids = [self.char2idx.get(c, 1) for c in text]
        while len(ids) < max_len:
            ids.append(0)
        return ids[:max_len]

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx, "vocab_size": self.vocab_size,
                       "max_len": self.max_len}, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["vocab_size"], data["max_len"])
        tok.char2idx = data["char2idx"]
        return tok


def build_model(config=None):
    return FakeDocumentModel(vocab_size=10000, embed_dim=128, max_len=256)
