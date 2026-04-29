"""
Multimodal inference pipeline.
Detects content type, routes to model, runs explainability, generates report.
"""

import os
import sys
import time
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference.content_router import detect_content_type, route_to_model


def load_system_config(config_path="configs/system_configs.yaml"):
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {"inference": {"gpu_enabled": False}, "storage": {}}


def run_full_pipeline(file_path: str = None, text_input: str = None,
                      model_dir: str = "saved_models",
                      config_path: str = "configs/system_configs.yaml",
                      model_config_path: str = "configs/model_configs.yaml",
                      content_type_hint: str = None) -> dict:
    """
    Full multimodal inference pipeline.

    Args:
        file_path: Path to uploaded file (image/video/audio/document)
        text_input: Raw text or URL for news analysis
        model_dir: Directory containing trained model checkpoints
        config_path: System config path
        model_config_path: Model config path

    Returns:
        Structured analysis result dict
    """
    start_time = time.time()
    sys_cfg = load_system_config(config_path)

    import torch
    device = "cuda" if (sys_cfg["inference"].get("gpu_enabled", False) and torch.cuda.is_available()) else "cpu"

    # Load model-specific config
    try:
        with open(model_config_path) as f:
            all_model_cfg = yaml.safe_load(f)
    except Exception:
        all_model_cfg = {}

    # Determine content type
    if text_input:
        content_type = "news"
        input_data = text_input
    elif file_path:
        content_type = content_type_hint or detect_content_type(file_path)
        input_data = file_path
    else:
        return {"error": "No input provided"}

    # Get model config for this content type
    cfg_key_map = {
        "image": "image_model", "video": "video_model",
        "audio": "audio_model", "news": "news_model", "document": "document_model"
    }
    model_cfg = all_model_cfg.get(cfg_key_map.get(content_type, ""), {})

    # Route to appropriate model
    result = route_to_model(content_type, input_data, model_dir, model_cfg, device,
                            config_path=config_path)

    # Add pipeline metadata
    result["processing_time_sec"] = round(time.time() - start_time, 3)
    result["file_path"] = file_path
    result["pipeline_version"] = "1.0.0"

    return result


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python multimodal_inference.py <file_path_or_text>")
        sys.exit(1)

    inp = sys.argv[1]
    if inp.startswith("http") or (not os.path.exists(inp) and len(inp) > 20):
        result = run_full_pipeline(text_input=inp)
    else:
        result = run_full_pipeline(file_path=inp)

    print(json.dumps(result, indent=2, default=str))
