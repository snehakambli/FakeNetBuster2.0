"""Document inference wrapper with tampered region visualization."""

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
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def run_inference(file_path: str, model_dir: str, config: dict = None, device: str = "cpu") -> dict:
    from ml_models.fake_documents.predict import predict

    model_path = os.path.join(model_dir, "document_model_best.pth")

    tokenizer_path = os.path.join(model_dir, "document_tokenizer.json")

    if not os.path.exists(model_path):
        return {
            "content_type": "document",
            "error": f"Model not found at {model_path}. Train the model first.",
            "prediction": "unknown",
            "confidence_score": 0.0,
        }

    result = predict(file_path, model_path,
                     tokenizer_path if os.path.exists(tokenizer_path) else None,
                     config, device)

    # ELA visualization
    ela_path, ela_map = _generate_ela_visualization(file_path)
    if ela_path:
        result["ela_visualization_path"] = ela_path
        result["ela_base64"] = _img_to_base64(ela_path)

    # Annotated document with fault markers
    annotated_path = _generate_annotated(file_path, ela_map, result)
    if annotated_path:
        result["annotated_path"] = annotated_path
        result["annotated_base64"] = _img_to_base64(annotated_path)

    return result


def _generate_ela_visualization(doc_path: str):
    """Generate Error Level Analysis visualization. Returns (path, ela_gray_map)."""
    import io
    from PIL import Image

    viz_dir = _get_viz_dir()
    os.makedirs(viz_dir, exist_ok=True)

    try:
        img = Image.open(doc_path).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        ela = np.abs(np.array(img, dtype=np.float32) -
                     np.array(compressed, dtype=np.float32))
        ela_scaled = (ela * 10).clip(0, 255).astype(np.uint8)

        ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
        ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)

        fname = Path(doc_path).stem + "_ela.jpg"
        out_path = os.path.join(viz_dir, fname)
        cv2.imwrite(out_path, ela_colored)
        return out_path, ela_gray
    except Exception:
        return "", None


def _generate_annotated(doc_path: str, ela_map, result: dict) -> str:
    """Draw fault markers on the document image based on ELA + anomalies."""
    viz_dir = _get_viz_dir()
    os.makedirs(viz_dir, exist_ok=True)

    img = cv2.imread(doc_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    annotated = img.copy()
    anomalies = result.get("detected_anomalies", [])
    fake_prob = result.get("fake_probability", 0.0)
    prediction = result.get("prediction", "unknown").upper()
    conf = result.get("confidence_score", 0.0)

    fault_count = 0

    # Use ELA map to find tampered regions
    if ela_map is not None:
        ela_resized = cv2.resize(ela_map, (w, h))
        # Threshold: regions with high ELA score are suspicious
        ela_score = float(ela_resized.mean())
        thresh_val = min(int(ela_score * 3 + 30), 200)
        _, binary = cv2.threshold(ela_resized, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

        for cnt in contours:
            if cv2.contourArea(cnt) < (h * w * 0.003):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            fault_count += 1
            color = (0, 140, 255) if fake_prob > 0.65 else (0, 180, 255)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

    # No legend strip — result shown in UI overlay
    final = annotated

    fname = Path(doc_path).stem + "_annotated.jpg"
    out_path = os.path.join(viz_dir, fname)
    cv2.imwrite(out_path, final)
    return out_path
