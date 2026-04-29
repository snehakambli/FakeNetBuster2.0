"""File handling utilities."""

import os
import uuid
import shutil
from pathlib import Path


ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png"},
    "video": {".mp4", ".mov", ".avi"},
    "audio": {".wav", ".mp3"},
    "document": {".pdf", ".jpg", ".jpeg", ".png"},
}


def generate_unique_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def save_upload(file_content: bytes, filename: str, upload_dir: str = "uploads") -> str:
    os.makedirs(upload_dir, exist_ok=True)
    unique_name = generate_unique_filename(filename)
    file_path = os.path.join(upload_dir, unique_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def validate_file_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    all_allowed = set()
    for exts in ALLOWED_EXTENSIONS.values():
        all_allowed.update(exts)
    return ext in all_allowed


def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass
