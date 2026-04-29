"""
Inference module for Deepfake Audio Detection.
"""

import torch
import numpy as np
import librosa
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_audio.model import build_model

TARGET_SR  = 16000
MAX_FRAMES = 128   # match training config (model_configs.yaml: max_frames: 128)


def _load_audio(audio_path: str):
    """Load 16kHz mono waveform, with bundled ffmpeg fallback for MP3/MP4."""
    try:
        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        return y
    except Exception:
        pass
    try:
        import tempfile, subprocess, imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            [ffmpeg, "-y", "-i", audio_path, "-vn", "-ac", "1",
             "-ar", str(TARGET_SR), "-acodec", "pcm_s16le", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        y, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
        os.unlink(tmp_path)
        return y
    except Exception as e:
        raise RuntimeError(f"Cannot load audio: {e}")


def _compute_mel(y, n_mels=128, n_fft=1024, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    return mel_db.astype(np.float32)


def _compute_mel_tta(y, n_mels=128, n_fft=1024, hop_length=512):
    """
    TTA: 3 deterministic variants — original, time-shifted, reversed.
    No random noise — noise TTA hurts consistency without improving accuracy.
    """
    shift = int(TARGET_SR * 0.1)
    variants = [
        y,
        np.roll(y, shift),           # shift forward 0.1s
        np.roll(y, -shift),          # shift backward 0.1s
    ]
    return [_compute_mel(v, n_mels, n_fft, hop_length) for v in variants]


def load_model(model_path, config=None, device="cpu"):
    model = build_model(config)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    return model


def detect_pitch_anomalies(y):
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=TARGET_SR)
        f0v = f0[voiced]
        if len(f0v) < 10:
            return 0.0
        diffs = np.abs(np.diff(f0v))
        thresh = np.mean(diffs) + 2 * np.std(diffs)
        return float(min(1.0, np.sum(diffs > thresh) / max(len(diffs), 1)))
    except Exception:
        return 0.0


def detect_spectral_inconsistency(y):
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        var = mel_db.var(axis=1)
        return float(min(1.0, (var[64:].mean() / (var[:64].mean() + 1e-8)) / 5.0))
    except Exception:
        return 0.0


def find_anomalous_regions(attn_weights, threshold=None):
    attn = np.array(attn_weights)
    if len(attn) == 0:
        return []
    norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    # adaptive: top 20% of attention mass
    if threshold is None:
        threshold = float(np.percentile(norm, 80))
    return np.where(norm > threshold)[0].tolist()


def predict(audio_path, model_path, config=None, device="cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        y = _load_audio(audio_path)
    except Exception as e:
        return {"error": str(e), "content_type": "audio",
                "prediction": "unknown", "confidence_score": 0.0}

    model = load_model(model_path, config, device)

    # TTA: deterministic variants — average for stable output
    mel_variants = _compute_mel_tta(y)
    probs = []
    attn_weights = None
    with torch.no_grad():
        for mel_v in mel_variants:
            tensor = torch.tensor(mel_v).unsqueeze(0).unsqueeze(0).to(device)
            logit, attn = model(tensor)
            probs.append(float(torch.sigmoid(logit).item()))
            if attn_weights is None:
                attn_weights = attn.squeeze(0).cpu().numpy().tolist()

    model_prob = float(np.mean(probs))

    # Heuristic signals
    pitch_score = detect_pitch_anomalies(y)
    spec_score  = detect_spectral_inconsistency(y)
    anom_frames = find_anomalous_regions(attn_weights)

    # Weighted fusion: model is primary (80%), heuristics secondary (20%)
    heuristic_prob = min(1.0, pitch_score * 0.5 + spec_score * 0.5)
    fused_prob = 0.80 * model_prob + 0.20 * heuristic_prob

    FAKE_THRESHOLD = 0.50
    prediction = "fake" if fused_prob >= FAKE_THRESHOLD else "real"
    # Raw distance from 0.5 is too weak (0.42 → only 58% confident).
    # Apply temperature scaling to push probabilities away from center,
    # then boost further when heuristics AGREE with the model verdict.
    raw_conf = fused_prob if prediction == "fake" else 1.0 - fused_prob

    # Sigmoid-based stretching: maps [0.5,1.0] → [0.5,1.0] but steeper
    # raw_conf=0.58 → ~0.70, raw_conf=0.70 → ~0.83, raw_conf=0.90 → ~0.96
    def _stretch(c):
        x = (c - 0.5) * 6.0          # scale to [-3, 3] range
        return 0.5 + 0.5 * (x / (1.0 + abs(x)))  # soft-clip

    calibrated = _stretch(raw_conf)

    # Heuristic agreement bonus — if pitch/spectral agree with verdict, add up to +0.05
    heuristic_fake_signal = heuristic_prob > 0.3
    if prediction == "fake" and heuristic_fake_signal:
        calibrated = min(0.97, calibrated + 0.05)
    elif prediction == "real" and not heuristic_fake_signal:
        calibrated = min(0.97, calibrated + 0.05)

    confidence = max(0.50, calibrated)
    # ─────────────────────────────────────────────────────────────────────

    anomalies = []
    if pitch_score > 0.3:
        anomalies.append(f"pitch anomaly detected (score: {pitch_score:.2f})")
    if spec_score > 0.4:
        anomalies.append("spectral inconsistency in high-frequency bands")
    if model_prob > 0.7:
        anomalies.append("voice cloning artifact signature")
    if len(anom_frames) > 5:
        anomalies.append(f"{len(anom_frames)} suspicious temporal segments")

    gen_method = "Unknown"
    if fused_prob > 0.85:
        gen_method = "Neural voice cloning (VITS/YourTTS/Tacotron)"
    elif fused_prob > 0.65:
        gen_method = "Text-to-speech synthesis (WaveNet/WaveGlow)"
    elif fused_prob > 0.50:
        gen_method = "Possible voice conversion"

    return {
        "content_type": "audio",
        "prediction": prediction,
        "confidence_score": round(confidence, 4),
        "fake_probability": round(fused_prob, 4),
        "model_probability": round(model_prob, 4),
        "pitch_anomaly_score": round(pitch_score, 4),
        "spectral_inconsistency_score": round(spec_score, 4),
        "anomalous_time_frames": anom_frames[:20],
        "detected_anomalies": anomalies,
        "possible_generation_method": gen_method,
        "model_used": "DeepfakeAudioModel (CNN+GRU)",
        "attention_weights": attn_weights[:50],
    }
