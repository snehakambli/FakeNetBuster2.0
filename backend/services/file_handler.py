"""
File handling service for uploads.
"""

import os
import uuid
import yaml
from pathlib import Path
from fastapi import UploadFile, HTTPException


def get_storage_config():
    try:
        with open("configs/system_configs.yaml") as f:
            return yaml.safe_load(f)["storage"]
    except Exception:
        return {
            "upload_dir": "uploads",
            "max_file_size_mb": 500,
            "allowed_image_types": ["jpg", "jpeg", "png"],
            "allowed_video_types": ["mp4", "mov", "avi"],
            "allowed_audio_types": ["wav", "mp3"],
            "allowed_document_types": ["pdf", "jpg", "png"],
        }


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi",
                      ".wav", ".mp3", ".pdf"}


async def save_upload_file(upload_file: UploadFile) -> dict:
    """Save uploaded file and return metadata."""
    cfg = get_storage_config()
    upload_dir = cfg.get("upload_dir", "uploads")
    max_size_mb = cfg.get("max_file_size_mb", 500)

    os.makedirs(upload_dir, exist_ok=True)

    ext = Path(upload_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {ALLOWED_EXTENSIONS}"
        )

    file_id = uuid.uuid4().hex
    unique_filename = f"{file_id}{ext}"
    file_path = os.path.join(upload_dir, unique_filename)

    content = await upload_file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > max_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max: {max_size_mb}MB"
        )

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": upload_file.filename,
        "file_path": file_path,
        "file_size_mb": round(size_mb, 3),
        "extension": ext,
    }
