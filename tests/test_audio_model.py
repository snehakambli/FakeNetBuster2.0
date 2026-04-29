"""Tests for deepfake audio model."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_models.deepfake_audio.model import DeepfakeAudioModel, build_model, SpectrogramCNN


class TestSpectrogramCNN:
    def test_output_shape(self):
        cnn = SpectrogramCNN(n_mels=128)
        x = torch.randn(2, 1, 128, 64)  # (B, 1, n_mels, T)
        with torch.no_grad():
            out = cnn(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 256  # feature dim


class TestDeepfakeAudioModel:
    def setup_method(self):
        self.model = build_model()
        self.model.eval()

    def test_forward_pass(self):
        x = torch.randn(2, 1, 128, 128)  # (B, 1, n_mels, T)
        with torch.no_grad():
            prob, attn = self.model(x)
        assert prob.shape == (2, 1)

    def test_output_range(self):
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            prob, _ = self.model(x)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_build_with_config(self):
        config = {"n_mels": 64, "gru_hidden": 128, "gru_layers": 1}
        model = build_model(config)
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            prob, _ = model(x)
        assert prob.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
