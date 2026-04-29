"""News inference wrapper — passes API keys from system config."""

import os
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _load_api_keys(config_path: str = "configs/system_configs.yaml") -> dict:
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("news_apis", {})
    except Exception:
        return {}


def run_inference(text_or_url: str, model_dir: str,
                  config: dict = None, device: str = "cpu",
                  config_path: str = "configs/system_configs.yaml") -> dict:
    from ml_models.fake_news.predict import predict

    model_path = os.path.join(model_dir, "news_model_best.pth")

    tokenizer_path = os.path.join(model_dir, "news_tokenizer.json")

    api_keys = _load_api_keys(config_path)

    return predict(
        text_or_url,
        model_path,
        tokenizer_path,
        config=config,
        device=device,
        api_keys=api_keys if api_keys else None,
    )
