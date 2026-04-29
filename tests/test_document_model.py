"""Tests for fake document detection model."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_models.fake_documents.model import FakeDocumentModel, build_model, DocumentTokenizer


class TestDocumentTokenizer:
    def setup_method(self):
        self.tokenizer = DocumentTokenizer(max_len=64)

    def test_encode_length(self):
        ids = self.tokenizer.encode("test document text", max_len=64)
        assert len(ids) == 64

    def test_padding(self):
        ids = self.tokenizer.encode("hi", max_len=32)
        assert ids[-1] == 0

    def test_unknown_char(self):
        ids = self.tokenizer.encode("@#$%", max_len=16)
        assert all(i >= 0 for i in ids)


class TestFakeDocumentModel:
    def setup_method(self):
        self.model = build_model()
        self.model.eval()

    def test_visual_only(self):
        img = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            prob = self.model(img)
        assert prob.shape == (2, 1)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_with_text(self):
        img = torch.randn(2, 3, 512, 512)
        tokens = torch.randint(0, 100, (2, 256))
        with torch.no_grad():
            prob = self.model(img, tokens)
        assert prob.shape == (2, 1)

    def test_output_range(self):
        img = torch.randn(4, 3, 512, 512)
        with torch.no_grad():
            prob = self.model(img)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_gradient_flow(self):
        model = build_model()
        img = torch.randn(1, 3, 512, 512)
        prob = model(img)
        prob.mean().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
