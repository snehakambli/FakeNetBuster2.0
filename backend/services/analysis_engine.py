"""
Analysis engine service - bridges API to ML inference pipeline.
"""

import os
import sys
import asyncio
from functools import partial
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from inference.multimodal_inference import run_full_pipeline
from inference.content_router import detect_content_type
from reports.report_generator import generate_report


def get_model_dir():
    try:
        with open("configs/system_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("storage", {}).get("model_dir", "saved_models")
    except Exception:
        return "saved_models"


def _run_file_pipeline(file_path: str, content_type_hint: str = None) -> dict:
    """Synchronous pipeline — runs in thread pool to avoid blocking the event loop."""
    model_dir = get_model_dir()

    if content_type_hint and content_type_hint in ["image", "video", "audio", "document"]:
        # Pass hint via file_path; content router will use it
        pass
    else:
        content_type_hint = None

    result = run_full_pipeline(
        file_path=file_path,
        model_dir=model_dir,
        config_path="configs/system_configs.yaml",
        model_config_path="configs/model_configs.yaml",
        content_type_hint=content_type_hint,
    )
    return generate_report(result, file_path=file_path)


def _run_text_pipeline(text_or_url: str) -> dict:
    """Synchronous pipeline — runs in thread pool."""
    model_dir = get_model_dir()
    result = run_full_pipeline(
        text_input=text_or_url,
        model_dir=model_dir,
        config_path="configs/system_configs.yaml",
        model_config_path="configs/model_configs.yaml",
    )
    return generate_report(result)


async def analyze_file(file_path: str, content_type_hint: str = None) -> dict:
    """Run file analysis in a thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_run_file_pipeline, file_path, content_type_hint)
    )


async def analyze_text(text_or_url: str) -> dict:
    """Run news analysis in a thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_run_text_pipeline, text_or_url)
    )
