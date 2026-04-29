"""
Inference module for Deepfake Image Detection.
Returns prediction, confidence, and anomaly details.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.deepfake_image.model import build_model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_tta_transforms(input_size=224):
    """5 TTA variants: original, hflip, slight crop x2, center crop."""
    base = [transforms.Resize((input_size, input_size))]
    to_tensor = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return [
        transforms.Compose(base + to_tensor),
        transforms.Compose([transforms.Resize((input_size, input_size)),
                            transforms.RandomHorizontalFlip(p=1.0)] + to_tensor),
        transforms.Compose([transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
                            transforms.CenterCrop(input_size)] + to_tensor),
        transforms.Compose([transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
                            transforms.CenterCrop(input_size)] + to_tensor),
        transforms.Compose([transforms.Resize((input_size + 20, input_size + 20)),
                            transforms.FiveCrop(input_size),
                            transforms.Lambda(lambda crops: crops[0])] + to_tensor),
    ]


def load_model(model_path, config=None, device="cpu"):
    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_gradcam(model, img_tensor, device):
    """Compute GradCAM heatmap for the last conv layer."""
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, inp, out):
        activations.append(out)
        out.register_hook(save_gradient)

    # Hook into last block of EfficientNet spatial features
    target_layer = model.spatial_features[-1]
    handle = target_layer.register_forward_hook(forward_hook)

    img_tensor = img_tensor.to(device).requires_grad_(True)
    output = torch.sigmoid(model(img_tensor))
    model.zero_grad()
    output.backward()
    handle.remove()

    if not gradients or not activations:
        return None

    grad = gradients[0].cpu().data.numpy()[0]       # (C, H, W)
    act = activations[0].cpu().data.numpy()[0]       # (C, H, W)
    weights = grad.mean(axis=(1, 2))                 # (C,)
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def detect_noise_inconsistency(img_np):
    """Detect noise pattern inconsistencies using SRM filters."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Simple Rich Model filter
    kernel = np.array([[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]], dtype=np.float32)
    residual = cv2.filter2D(gray, -1, kernel)
    noise_std = residual.std()
    return float(noise_std)


def detect_color_inconsistency(img_np):
    """Detect color blending mismatches in face regions."""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, w = hsv.shape[:2]
    # Compare center vs border saturation variance
    center = hsv[h//4:3*h//4, w//4:3*w//4, 1]
    border_top = hsv[:h//4, :, 1]
    border_bot = hsv[3*h//4:, :, 1]
    center_var = center.var()
    border_var = np.concatenate([border_top.flatten(), border_bot.flatten()]).var()
    inconsistency = abs(center_var - border_var) / (border_var + 1e-8)
    return float(inconsistency)


def classify_generation_method(confidence, noise_score, color_score):
    """Heuristic classification of likely AI generation method."""
    if confidence > 0.85:
        if noise_score < 5.0:
            return "Diffusion Model (low noise residual)"
        elif color_score > 0.5:
            return "GAN-based face swap (color boundary mismatch)"
        else:
            return "GAN-based generation (frequency artifacts)"
    elif confidence > 0.6:
        return "Possible GAN or neural rendering"
    return "Unknown / Low confidence"


def predict(image_path, model_path, config=None, device="cpu"):
    """
    Run full inference on an image with test-time augmentation (TTA).
    Returns structured result dict.
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = load_model(model_path, config, device)

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    input_size = config.get("input_size", 224) if config else 224
    anomalies = []

    # TTA: average predictions across 5 augmented views
    tta_transforms = get_tta_transforms(input_size)
    probs = []
    with torch.no_grad():
        for tfm in tta_transforms:
            tensor = tfm(img).unsqueeze(0).to(device)
            logit = model(tensor)
            probs.append(torch.sigmoid(logit).item())
    prob = float(np.mean(probs))

    img_tensor_grad = get_transform(input_size)(img).unsqueeze(0)
    cam = compute_gradcam(model, img_tensor_grad, device)

    # Anomaly analysis
    noise_score = detect_noise_inconsistency(img_np)
    color_score = detect_color_inconsistency(img_np)

    prediction = "fake" if prob > 0.5 else "real"
    confidence = prob if prob > 0.5 else 1.0 - prob
    if noise_score > 8.0:
        anomalies.append("Noise pattern inconsistency detected")
    if color_score > 0.3:
        anomalies.append("Color blending mismatch at face boundary")
    if prob > 0.6:
        anomalies.append("GAN frequency fingerprint detected")
    if prob > 0.8:
        anomalies.append("High-confidence manipulation signature")

    gen_method = classify_generation_method(prob, noise_score, color_score)

    result = {
        "content_type": "image",
        "prediction": prediction,
        "confidence_score": round(confidence, 4),
        "fake_probability": round(prob, 4),
        "detected_anomalies": anomalies,
        "noise_inconsistency_score": round(noise_score, 4),
        "color_inconsistency_score": round(color_score, 4),
        "possible_generation_method": gen_method,
        "gradcam_available": cam is not None,
        "gradcam": cam.tolist() if cam is not None else None,
        "model_used": "DeepfakeImageModel (EfficientNet-B4 + Frequency + Noise)",
    }
    return result


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_path> <model_path>")
        sys.exit(1)
    result = predict(sys.argv[1], sys.argv[2])
    result.pop("gradcam", None)
    print(json.dumps(result, indent=2))
