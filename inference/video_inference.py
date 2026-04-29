"""Video inference wrapper."""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def run_inference(file_path: str, model_dir: str, config: dict = None, device: str = "cpu") -> dict:
    from ml_models.deepfake_video.predict import predict

    model_path = os.path.join(model_dir, "video_model_best.pth")

    if not os.path.exists(model_path):
        return {
            "content_type": "video",
            "error": f"Model not found at {model_path}. Train the model first.",
            "prediction": "unknown",
            "confidence_score": 0.0,
        }

    result = predict(file_path, model_path, config, device)

    # Extract top suspicious frames as base64 for inline display
    # Do this BEFORE removing window_analysis from result
    if result.get("suspicious_frames"):
        result["suspicious_frames_b64"] = _extract_frames_b64(
            file_path, result["suspicious_frames"], result.get("window_analysis", [])
        )

    result.pop("window_analysis", None)
    return result


def _extract_frames_b64(video_path: str, suspicious_frames: list, window_analysis: list) -> list:
    """
    Extract up to 4 most suspicious frames and return as base64 JPEG strings.
    Picks frames from the highest-probability windows first.
    """
    import cv2
    import base64
    import numpy as np

    # Score each suspicious frame by the window probability it came from
    frame_scores = {}
    for w in window_analysis:
        prob = w.get("prob", 0)
        start = w.get("start_frame", 0)
        attn = w.get("attention", [])
        for i, a in enumerate(attn):
            fn = start + i * 5
            if fn in suspicious_frames:
                frame_scores[fn] = max(frame_scores.get(fn, 0), prob * a)

    # Fall back to raw order if no scores
    if frame_scores:
        ranked = sorted(frame_scores, key=frame_scores.get, reverse=True)
    else:
        ranked = suspicious_frames

    # Pick top 4, spread across the video
    selected = []
    seen_regions = []
    for fn in ranked:
        # Avoid frames too close together (within 30 frames)
        if all(abs(fn - s) > 30 for s in seen_regions):
            selected.append(fn)
            seen_regions.append(fn)
        if len(selected) == 4:
            break

    # If fewer than 4, fill from remaining suspicious frames
    for fn in ranked:
        if fn not in selected and len(selected) < 4:
            selected.append(fn)

    selected = sorted(selected[:4])

    cap = cv2.VideoCapture(video_path)
    results = []
    for fn in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (320, 240))
        # Draw frame number label
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {fn}", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode('utf-8')
        results.append({
            "frame": fn,
            "data": f"data:image/jpeg;base64,{b64}"
        })
    cap.release()
    return results
