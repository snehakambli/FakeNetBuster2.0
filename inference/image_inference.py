"""Image inference wrapper with explainability."""

import os
import sys
import base64
import numpy as np
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _get_viz_dir():
    try:
        import yaml
        with open("configs/system_configs.yaml") as f:
            return yaml.safe_load(f)["explainability"]["viz_dir"]
    except Exception:
        return "reports/visualizations"


def _img_to_base64(path: str) -> str:
    """Read an image file and return a base64 data URI."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def run_inference(file_path: str, model_dir: str, config: dict = None, device: str = "cpu") -> dict:
    from ml_models.deepfake_image.predict import predict

    model_path = os.path.join(model_dir, "image_model_best.pth")

    if not os.path.exists(model_path):
        return {
            "content_type": "image",
            "error": f"Model not found at {model_path}. Train the model first.",
            "prediction": "unknown",
            "confidence_score": 0.0,
        }

    result = predict(file_path, model_path, config, device)

    # Generate GradCAM overlay + annotated fault image
    if result.get("gradcam_available") and result.get("gradcam"):
        viz_path = _save_gradcam(file_path, result["gradcam"])
        if viz_path:
            result["visualization_path"] = viz_path
            result["visualization_base64"] = _img_to_base64(viz_path)

        annotated_path = _save_annotated(file_path, result["gradcam"], result)
        if annotated_path:
            result["annotated_path"] = annotated_path
            result["annotated_base64"] = _img_to_base64(annotated_path)

    result.pop("gradcam", None)
    return result


def _save_gradcam(image_path: str, cam_data: list) -> str:
    """Overlay GradCAM heatmap on original image and save."""
    viz_dir = _get_viz_dir()
    os.makedirs(viz_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return ""

    cam = np.array(cam_data, dtype=np.float32)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    fname = Path(image_path).stem + "_gradcam.jpg"
    out_path = os.path.join(viz_dir, fname)
    cv2.imwrite(out_path, overlay)
    return out_path


def _save_annotated(image_path: str, cam_data: list, result: dict) -> str:
    """Draw fault markers and labels on the image based on detected anomalies."""
    viz_dir = _get_viz_dir()
    os.makedirs(viz_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    cam = np.array(cam_data, dtype=np.float32)
    cam_resized = cv2.resize(cam, (w, h))

    # Subtle heatmap blend — keep image readable
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    annotated = cv2.addWeighted(img, 0.72, heatmap, 0.28, 0)

    anomalies = result.get("detected_anomalies", [])
    fake_prob = result.get("fake_probability", 0.0)
    prediction = result.get("prediction", "unknown").upper()
    conf = result.get("confidence_score", 0.0)

    # Find high-activation contours
    threshold = 0.65
    binary = (cam_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    fault_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < (h * w * 0.008):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        fault_count += 1
        color = (0, 0, 230) if fake_prob > 0.65 else (0, 140, 255)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        label = f"#{fault_count}"
        lx = x + bw + 4 if x + bw + 40 < w else x - 24
        ly = y + bh // 2 + 5
        cv2.putText(annotated, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # No legend strip — result shown cleanly in UI
    final = annotated

    fname = Path(image_path).stem + "_annotated.jpg"
    out_path = os.path.join(viz_dir, fname)
    cv2.imwrite(out_path, final)
    return out_path
