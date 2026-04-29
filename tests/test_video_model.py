"""Tests for deepfake video model."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_models.deepfake_video.model import DeepfakeVideoModel, build_model, FrameFeatureExtractor


class TestFrameFeatureExtractor:
    def test_output_shape(self):
        extractor = FrameFeatureExtractor()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = extractor(x)
        assert out.shape == (4, 512)


class TestDeepfakeVideoModel:
    def setup_method(self):
        self.model = build_model()
        self.model.eval()

    def test_forward_pass(self):
        x = torch.randn(2, 8, 3, 224, 224)  # (B, T, C, H, W)
        with torch.no_grad():
            prob, attn = self.model(x)
        assert prob.shape == (2, 1)
        assert attn.shape == (2, 8)

    def test_output_range(self):
        x = torch.randn(1, 8, 3, 224, 224)
        with torch.no_grad():
            prob, attn = self.model(x)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_attention_weights_sum(self):
        x = torch.randn(1, 8, 3, 224, 224)
        with torch.no_grad():
            _, attn = self.model(x)
        # Attention weights should sum to ~1
        assert abs(attn.sum().item() - 1.0) < 0.1

    def test_single_frame(self):
        x = torch.randn(1, 1, 3, 224, 224)
        with torch.no_grad():
            prob, attn = self.model(x)
        assert prob.shape == (1, 1)

    def test_build_with_config(self):
        config = {"lstm_hidden": 256, "lstm_layers": 1}
        model = build_model(config)
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
