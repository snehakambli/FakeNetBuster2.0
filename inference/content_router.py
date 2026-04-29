"""
Content type detection and routing to appropriate inference module.
"""

import os
import mimetypes
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".flac"}
DOCUMENT_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


def detect_content_type(file_path: str) -> str:
    """
    Detect content type from file extension and MIME type.
    Returns: 'image' | 'video' | 'audio' | 'document' | 'news' | 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    mime, _ = mimetypes.guess_type(file_path)

    if ext in VIDEO_EXTS or (mime and mime.startswith("video/")):
        return "video"
    if ext in AUDIO_EXTS or (mime and mime.startswith("audio/")):
        return "audio"
    if ext == ".pdf":
        return "document"
    if ext in IMAGE_EXTS:
        # Distinguish between photo and document based on filename hints
        fname = Path(file_path).stem.lower()
        doc_keywords = ["aadhar", "aadhaar", "id", "card", "certificate",
                        "passport", "license", "document", "doc", "cert"]
        if any(kw in fname for kw in doc_keywords):
            return "document"
        return "image"
    return "unknown"


def route_to_model(content_type: str, file_path: str, model_dir: str,
                   config: dict = None, device: str = "cpu",
                   config_path: str = "configs/system_configs.yaml") -> dict:
    """
    Route file to appropriate inference module based on content type.
    Returns raw inference result dict.
    """
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    if content_type == "image":
        from inference.image_inference import run_inference
        return run_inference(file_path, model_dir, config, device)

    elif content_type == "video":
        from inference.video_inference import run_inference
        return run_inference(file_path, model_dir, config, device)

    elif content_type == "audio":
        from inference.audio_inference import run_inference
        return run_inference(file_path, model_dir, config, device)

    elif content_type == "document":
        from inference.document_inference import run_inference
        return run_inference(file_path, model_dir, config, device)

    elif content_type == "news":
        from inference.news_inference import run_inference
        return run_inference(file_path, model_dir, config, device, config_path=config_path)

    else:
        return {
            "error": f"Unsupported content type: {content_type}",
            "content_type": "unknown"
        }
