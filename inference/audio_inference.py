"""Audio inference wrapper with spectrogram visualization."""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def run_inference(file_path: str, model_dir: str, config: dict = None, device: str = "cpu") -> dict:
    from ml_models.deepfake_audio.predict import predict

    model_path = os.path.join(model_dir, "audio_model_best.pth")

    if not os.path.exists(model_path):
        return {
            "content_type": "audio",
            "error": f"Model not found at {model_path}. Train the model first.",
            "prediction": "unknown",
            "confidence_score": 0.0,
        }

    result = predict(file_path, model_path, config, device)

    # Generate spectrogram visualization with anomaly highlights
    viz_path = _generate_spectrogram_viz(file_path, result.get("anomalous_time_frames", []))
    if viz_path:
        result["spectrogram_visualization_path"] = viz_path

    result.pop("attention_weights", None)
    return result


def _generate_spectrogram_viz(audio_path: str, anomalous_frames: list) -> str:
    """Generate mel-spectrogram image with highlighted anomalous regions."""
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return ""

    try:
        import yaml
        with open("configs/system_configs.yaml") as f:
            sys_cfg = yaml.safe_load(f)
        viz_dir = sys_cfg["explainability"]["viz_dir"]
    except Exception:
        viz_dir = "reports/visualizations"

    os.makedirs(viz_dir, exist_ok=True)

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                 x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title("Mel-Spectrogram (Red = Suspicious Regions)")

        # Highlight anomalous time frames
        hop_length = 512
        for frame_idx in anomalous_frames:
            time_sec = frame_idx * hop_length / sr
            rect = patches.Rectangle((time_sec, 0), hop_length / sr, sr / 2,
                                      linewidth=1, edgecolor="red",
                                      facecolor="red", alpha=0.3)
            ax.add_patch(rect)

        plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
        plt.tight_layout()

        fname = Path(audio_path).stem + "_spectrogram.png"
        out_path = os.path.join(viz_dir, fname)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        return ""
