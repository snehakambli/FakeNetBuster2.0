"""Tests for fake news detection model."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_models.fake_news.model import FakeNewsModel, build_model, SimpleTokenizer


class TestSimpleTokenizer:
    def setup_method(self):
        self.tokenizer = SimpleTokenizer(vocab_size=1000, max_len=64)
        self.tokenizer.fit(["this is a test sentence", "fake news detection works"])

    def test_encode_length(self):
        ids, mask = self.tokenizer.encode("hello world test", max_len=64)
        assert len(ids) == 64
        assert len(mask) == 64

    def test_padding(self):
        ids, mask = self.tokenizer.encode("short", max_len=32)
        assert ids[-1] == 0  # padded with 0

    def test_truncation(self):
        long_text = " ".join(["word"] * 100)
        ids, mask = self.tokenizer.encode(long_text, max_len=32)
        assert len(ids) == 32

    def test_unknown_token(self):
        ids, mask = self.tokenizer.encode("xyzunknownword123", max_len=16)
        assert 1 in ids  # UNK token


class TestFakeNewsModel:
    def setup_method(self):
        self.model = build_model({"vocab_size": 1000, "embed_dim": 64,
                                   "num_heads": 4, "num_layers": 2,
                                   "ff_dim": 128, "max_length": 64})
        self.model.eval()

    def test_forward_pass(self):
        ids = torch.randint(0, 1000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        with torch.no_grad():
            prob, attn = self.model(ids, mask)
        assert prob.shape == (2, 1)

    def test_output_range(self):
        ids = torch.randint(0, 1000, (4, 64))
        with torch.no_grad():
            prob, _ = self.model(ids)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_with_padding_mask(self):
        ids = torch.randint(0, 1000, (2, 64))
        mask = torch.zeros(2, 64, dtype=torch.long)
        mask[:, :10] = 1  # only first 10 tokens are real
        with torch.no_grad():
            prob, _ = self.model(ids, mask)
        assert prob.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
