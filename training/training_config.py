"""
Centralized training configuration loader.
"""

import yaml
import os


def load_config(config_path="configs/model_configs.yaml",
                training_config_path="configs/training_configs.yaml"):
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def get_model_config(modality, config_path="configs/model_configs.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    key_map = {
        "image": "image_model",
        "video": "video_model",
        "audio": "audio_model",
        "news": "news_model",
        "document": "document_model",
    }
    return cfg.get(key_map.get(modality, modality), {})
