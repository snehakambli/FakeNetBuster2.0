"""
Analysis Report Generator.
Produces structured JSON reports from inference results.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path


REPORT_TEMPLATES = {
    "image": "reports/templates/image_report_template.json",
    "video": "reports/templates/video_report_template.json",
    "audio": "reports/templates/audio_report_template.json",
    "news": "reports/templates/news_report_template.json",
    "document": "reports/templates/document_report_template.json",
}


def load_template(content_type: str) -> dict:
    template_path = REPORT_TEMPLATES.get(content_type)
    if template_path and os.path.exists(template_path):
        with open(template_path) as f:
            return json.load(f)
    return {}


def generate_report(inference_result: dict, file_path: str = None,
                    save_dir: str = "reports/generated") -> dict:
    """
    Generate a structured analysis report from inference results.

    Args:
        inference_result: Raw output from inference pipeline
        file_path: Original uploaded file path
        save_dir: Directory to save report JSON

    Returns:
        Complete structured report dict
    """
    os.makedirs(save_dir, exist_ok=True)

    content_type = inference_result.get("content_type", "unknown")
    template = load_template(content_type)

    report_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    prediction = inference_result.get("prediction", "unknown")
    confidence = inference_result.get("confidence_score", 0.0)
    fake_prob = inference_result.get("fake_probability", 0.0)

    # Risk level classification — aligned with fake threshold of 0.50
    if fake_prob >= 0.85:
        risk_level = "CRITICAL"
    elif fake_prob >= 0.65:
        risk_level = "HIGH"
    elif fake_prob >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    report = {
        "report_id": report_id,
        "timestamp": timestamp,
        "content_type": content_type,
        "file_path": file_path or inference_result.get("file_path", ""),
        "prediction": prediction,
        "confidence_score": confidence,
        "fake_probability": fake_prob,
        "risk_level": risk_level,
        "model_used": inference_result.get("model_used", "Unknown"),
        "possible_generation_method": inference_result.get("possible_generation_method", "Unknown"),
        "detected_anomalies": inference_result.get("detected_anomalies", []),
        "signals": inference_result.get("signals", []),
        "related_links": inference_result.get("related_links", []),
        "analysis_details": _build_analysis_details(content_type, inference_result),
        "processing_time_sec": inference_result.get("processing_time_sec", 0),
        "visualizations": _collect_visualizations(inference_result),
        "inline_images": _collect_inline_images(inference_result),
        "summary": _generate_summary(prediction, confidence, risk_level,
                                     inference_result.get("detected_anomalies", [])),
        "error": inference_result.get("error"),
    }

    # Save report to disk
    report_path = os.path.join(save_dir, f"{report_id}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    report["report_path"] = report_path
    return report


def _build_analysis_details(content_type: str, result: dict) -> dict:
    """Build content-type-specific analysis details."""
    if content_type == "image":
        return {
            "noise_inconsistency_score": result.get("noise_inconsistency_score"),
            "color_inconsistency_score": result.get("color_inconsistency_score"),
            "gradcam_available": result.get("gradcam_available", False),
            "visualization_path": result.get("visualization_path"),
            "annotated_path": result.get("annotated_path"),
        }
    elif content_type == "video":
        return {
            "suspicious_frames": result.get("suspicious_frames", []),
            "suspicious_frames_b64": result.get("suspicious_frames_b64", []),
            "suspicious_timestamps_sec": result.get("suspicious_timestamps_sec", []),
            "total_frames_analyzed": result.get("total_frames_analyzed", 0),
            "video_duration_sec": result.get("video_duration_sec", 0),
            "fps": result.get("fps", 0),
            "max_window_probability": result.get("max_window_probability", 0),
            "manipulated_regions": result.get("manipulated_regions", []),
            "timeline_visualization_path": result.get("timeline_visualization_path"),
        }
    elif content_type == "audio":
        return {
            "pitch_anomaly_score": result.get("pitch_anomaly_score"),
            "spectral_inconsistency_score": result.get("spectral_inconsistency_score"),
            "anomalous_time_frames": result.get("anomalous_time_frames", []),
            "spectrogram_shape": result.get("spectrogram_shape"),
            "spectrogram_visualization_path": result.get("spectrogram_visualization_path"),
        }
    elif content_type == "news":
        return {
            "source_url": result.get("source_url"),
            "word_count": result.get("word_count"),
            "suspicious_tokens": result.get("suspicious_tokens", []),
            "text_features": result.get("text_features", {}),
            "text_preview": result.get("text_preview", ""),
            "fact_check_apis": result.get("fact_check_apis", {}),
            "fact_check_results": result.get("fact_check_results", []),
            "cross_reference_count": result.get("cross_reference_count", 0),
            "claimbuster_score": result.get("claimbuster_score"),
            "ai_verdict": result.get("ai_verdict"),
        }
    elif content_type == "document":
        return {
            "ela_score": result.get("ela_score"),
            "font_inconsistency_score": result.get("font_inconsistency_score"),
            "seal_regions_detected": result.get("seal_regions_detected", 0),
            "tampered_regions": result.get("tampered_regions", []),
            "ocr_text_extracted": result.get("ocr_text_extracted", False),
            "ocr_preview": result.get("ocr_preview", ""),
            "ela_visualization_path": result.get("ela_visualization_path"),
            "annotated_path": result.get("annotated_path"),
        }
    return {}


def _collect_visualizations(result: dict) -> list:
    viz_keys = [
        "visualization_path", "annotated_path",
        "timeline_visualization_path",
        "spectrogram_visualization_path",
        "ela_visualization_path", "ela_base64",
    ]
    return [result[k] for k in viz_keys if result.get(k)]


def _collect_inline_images(result: dict) -> dict:
    """Collect base64-encoded images for inline display in reports."""
    keys = {
        "annotated_base64": "Annotated Image (faults marked)",
        "visualization_base64": "GradCAM Heatmap",
        "ela_base64": "Error Level Analysis",
        "spectrogram_base64": "Spectrogram Anomalies",
        "timeline_base64": "Suspicious Frame Timeline",
    }
    return {label: result[key] for key, label in keys.items() if result.get(key)}


def _generate_summary(prediction: str, confidence: float,
                       risk_level: str, anomalies: list) -> str:
    if prediction == "fake":
        anomaly_str = (f" Key anomalies: {', '.join(anomalies[:3])}."
                       if anomalies else "")
        return (f"Content classified as FAKE with {confidence*100:.1f}% confidence. "
                f"Risk level: {risk_level}.{anomaly_str}")
    else:
        return (f"Content classified as REAL with {confidence*100:.1f}% confidence. "
                f"No significant manipulation detected.")


def get_report(report_id: str, save_dir: str = "reports/generated") -> dict:
    """Load a previously generated report by ID."""
    report_path = os.path.join(save_dir, f"{report_id}.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        return json.load(f)


def list_reports(save_dir: str = "reports/generated", limit: int = 50) -> list:
    """List all generated reports, newest first."""
    if not os.path.exists(save_dir):
        return []
    reports = []
    for fname in sorted(os.listdir(save_dir), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(save_dir, fname)) as f:
                    r = json.load(f)
                reports.append({
                    "report_id": r.get("report_id"),
                    "timestamp": r.get("timestamp"),
                    "content_type": r.get("content_type"),
                    "prediction": r.get("prediction"),
                    "confidence_score": r.get("confidence_score"),
                    "risk_level": r.get("risk_level"),
                    "file_path": r.get("file_path"),
                })
            except Exception:
                continue
    return reports[:limit]
