"""
Fake News Detection Model
Architecture: Custom Transformer with positional encoding
Detects: misleading claims, clickbait patterns, semantic contradictions
Supports: multilingual input via language detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class FakeNewsModel(nn.Module):
    """
    Custom Transformer for fake news detection.
    Input: token ids (B, seq_len)
    Output: (B, 1) fake probability
    """
    def __init__(self, vocab_size=30000, embed_dim=256, num_heads=8,
                 num_layers=4, ff_dim=512, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        # attention_mask: 1 for real tokens, 0 for padding
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)  # True where padded

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        all_attn = []
        for block in self.transformer_blocks:
            x, attn = block(x, padding_mask)
            all_attn.append(attn)

        # Pool over sequence dimension
        x = x.transpose(1, 2)  # (B, embed_dim, seq_len)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        return self.classifier(x), all_attn[-1]

    def get_attention_weights(self, input_ids, attention_mask=None):
        """Return attention weights for explainability."""
        _, attn = self.forward(input_ids, attention_mask)
        return attn


class SimpleTokenizer:
    """
    Character/word-level tokenizer for multilingual support.
    Builds vocabulary from training data.
    """
    def __init__(self, vocab_size=30000, max_len=512):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.fitted = False

    def fit(self, texts):
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        for word, _ in word_counts.most_common(self.vocab_size - 4):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.fitted = True

    def encode(self, text, max_len=None):
        max_len = max_len or self.max_len
        words = ["<CLS>"] + text.lower().split() + ["<SEP>"]
        ids = [self.word2idx.get(w, 1) for w in words]
        if len(ids) > max_len:
            ids = ids[:max_len]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            mask.append(0)
        return ids, mask

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "vocab_size": self.vocab_size,
                       "max_len": self.max_len}, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["vocab_size"], data["max_len"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        tok.fitted = True
        return tok


def build_model(config=None):
    if config is None:
        config = {}
    return FakeNewsModel(
        vocab_size=config.get("vocab_size", 30000),
        embed_dim=config.get("embed_dim", 256),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 4),
        ff_dim=config.get("ff_dim", 512),
        max_len=config.get("max_length", 512),
    )
