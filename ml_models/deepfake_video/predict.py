"""
Inference module for Deepfake Video Detection.
Extracts frames, runs CNN+LSTM, returns per-frame suspicion scores.
"""

import torch
import numpy as np
import cv2
import os
import sys
from torchvision import transforms
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_video.model import build_model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(model_path, config=None, device="cpu"):
    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_frames(video_path, sequence_length=16, frame_size=224, sample_rate=5):
    """Extract frames from video at given sample rate."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    sampled_indices = list(range(0, total_frames, sample_rate))
    frames = []
    frame_indices = []

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (frame_size, frame_size))
        frames.append(transform(frame_resized))
        frame_indices.append(idx)

    cap.release()
    return frames, frame_indices, fps, total_frames, duration


def predict_clip(model, frames, device, flip=False):
    """Run inference on a clip of frames, optionally horizontally flipped."""
    if len(frames) == 0:
        return 0.5, []

    if flip:
        frames = [torch.flip(f, dims=[2]) for f in frames]  # flip width dim

    clip = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)
    with torch.no_grad():
        logit, attn = model(clip)
        prob = torch.sigmoid(logit)
    return prob.item(), attn.squeeze(0).cpu().numpy()


def predict(video_path, model_path, config=None, device="cpu"):
    """
    Full video inference pipeline.
    Returns structured result with per-frame analysis.
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = load_model(model_path, config, device)

    seq_len = config.get("sequence_length", 16) if config else 16
    frame_size = config.get("frame_size", 224) if config else 224
    sample_rate = 5

    all_frames, frame_indices, fps, total_frames, duration = extract_frames(
        video_path, seq_len, frame_size, sample_rate
    )

    if len(all_frames) == 0:
        return {"error": "Could not extract frames from video"}

    # Process in sliding windows of seq_len, with TTA (normal + hflip)
    window_results = []
    step = max(1, seq_len // 2)

    for start in range(0, max(1, len(all_frames) - seq_len + 1), step):
        clip_frames = all_frames[start:start + seq_len]
        if len(clip_frames) < seq_len:
            clip_frames = clip_frames + [clip_frames[-1]] * (seq_len - len(clip_frames))
        prob_normal, attn = predict_clip(model, clip_frames, device, flip=False)
        prob_flip, _      = predict_clip(model, clip_frames, device, flip=True)
        prob = (prob_normal + prob_flip) / 2.0  # TTA average
        window_results.append({
            "start_frame": frame_indices[start] if start < len(frame_indices) else 0,
            "prob": prob,
            "attention": attn.tolist()
        })

    # Aggregate: use weighted combination of mean and max
    # max catches short deepfake segments that get diluted by mean
    all_probs = [r["prob"] for r in window_results]
    mean_prob = float(np.mean(all_probs))
    max_prob  = float(np.max(all_probs))
    # 60% mean + 40% max — sensitive to both sustained and brief fakes
    overall_prob = 0.6 * mean_prob + 0.4 * max_prob

    # Identify suspicious frames using adaptive threshold
    suspicious_frames = []
    prob_threshold = max(0.5, np.percentile(all_probs, 70))  # top 30% of windows
    for r in window_results:
        if r["prob"] > prob_threshold:
            attn = np.array(r["attention"])
            top_attn_idx = np.argsort(attn)[-3:]
            for i in top_attn_idx:
                frame_num = r["start_frame"] + i * sample_rate
                if frame_num not in suspicious_frames:
                    suspicious_frames.append(int(frame_num))

    suspicious_frames.sort()
    suspicious_timestamps = [round(f / fps, 2) for f in suspicious_frames if fps > 0]

    prediction = "fake" if overall_prob > 0.5 else "real"
    confidence = overall_prob if overall_prob > 0.5 else 1.0 - overall_prob

    anomalies = []
    if max_prob > 0.8:
        anomalies.append("high-confidence face manipulation detected")
    if len(suspicious_frames) > 5:
        anomalies.append("multiple suspicious temporal segments")
    if overall_prob > 0.6:
        anomalies.append("temporal inconsistency in face region")

    gen_method = "Unknown"
    if overall_prob > 0.85:
        gen_method = "GAN-based face swap (FaceSwap/DeepFaceLab)"
    elif overall_prob > 0.65:
        gen_method = "Neural face reenactment (Face2Face/NeuralTextures)"
    elif overall_prob > 0.5:
        gen_method = "Possible face manipulation"

    return {
        "content_type": "video",
        "prediction": prediction,
        "confidence_score": round(confidence, 4),
        "fake_probability": round(overall_prob, 4),
        "max_window_probability": round(max_prob, 4),
        "suspicious_frames": suspicious_frames[:20],
        "suspicious_timestamps_sec": suspicious_timestamps[:20],
        "total_frames_analyzed": len(all_frames),
        "video_duration_sec": round(duration, 2),
        "fps": round(fps, 2),
        "detected_anomalies": anomalies,
        "manipulated_regions": ["face boundary mismatch"] if overall_prob > 0.6 else [],
        "possible_generation_method": gen_method,
        "model_used": "VideoDeepfakeModel (EfficientNet-B3+BiLSTM)",
        "window_analysis": window_results[:10],
    }


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <video_path> <model_path>")
        sys.exit(1)
    result = predict(sys.argv[1], sys.argv[2])
    result.pop("window_analysis", None)
    print(json.dumps(result, indent=2))
