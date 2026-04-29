"""Tests for deepfake image model."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_models.deepfake_image.model import DeepfakeImageModel, build_model, ResBlock as ResidualBlock


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ResidualBlock(32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert out.shape == (1, 32, 16, 16)


class TestDeepfakeImageModel:
    def setup_method(self):
        self.model = build_model()
        self.model.eval()

    def test_forward_pass(self):
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (2, 1)

    def test_output_range(self):
        x = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            out = torch.sigmoid(self.model(x))  # model outputs raw logits
        assert (out >= 0).all() and (out <= 1).all()

    def test_batch_size_1(self):
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (1, 1)

    def test_feature_maps(self):
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feat = self.model.get_feature_maps(x)
        assert feat.dim() == 4

    def test_build_model_with_config(self):
        config = {"dropout": 0.3}
        model = build_model(config)
        assert model is not None

    def test_gradient_flow(self):
        model = build_model()
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
