"""Entry point for image model training."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from ml_models.deepfake_image.train import train

if __name__ == "__main__":
    train(
        config_path="configs/model_configs.yaml",
        training_config_path="configs/training_configs.yaml",
        data_root="datasets/images",
        save_dir="saved_models"
    )
