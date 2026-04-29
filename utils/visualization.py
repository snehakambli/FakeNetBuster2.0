"""
Visualization utilities for explainability outputs.
GradCAM, spectrogram heatmaps, document anomaly highlights.
"""

import os
import numpy as np
import cv2
from pathlib import Path


def apply_gradcam_overlay(image_path: str, cam: np.ndarray,
                          alpha: float = 0.4, output_path: str = None) -> str:
    """Apply GradCAM heatmap overlay on image."""
    img = cv2.imread(image_path)
    if img is None:
        return ""

    cam_resized = cv2.resize(cam.astype(np.float32), (img.shape[1], img.shape[0]))
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"reports/visualizations/{stem}_gradcam.jpg"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    return output_path


def draw_bounding_boxes(image_path: str, regions: list,
                        output_path: str = None) -> str:
    """Draw bounding boxes around suspicious regions."""
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    for i, region in enumerate(regions):
        if isinstance(region, dict):
            x1 = int(region.get("x1", 0) * w)
            y1 = int(region.get("y1", 0) * h)
            x2 = int(region.get("x2", 1) * w)
            y2 = int(region.get("y2", 1) * h)
        else:
            # Default: highlight center region
            x1, y1 = w // 4, h // 4
            x2, y2 = 3 * w // 4, 3 * h // 4

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"Suspicious {i+1}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"reports/visualizations/{stem}_annotated.jpg"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def generate_confidence_bar(confidence: float, prediction: str) -> dict:
    """Generate data for confidence visualization bar."""
    color = "#ef4444" if prediction == "fake" else "#22c55e"
    return {
        "value": round(confidence * 100, 1),
        "color": color,
        "label": f"{prediction.upper()} ({confidence*100:.1f}%)"
    }
