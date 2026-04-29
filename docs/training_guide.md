# FakeNetBuster Training Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

GPU recommended. CPU training is supported but slow for video/image models.

---

## Dataset Preparation

### 1. Image Dataset (FFHQ + StyleGAN)

```
datasets/images/
├── train/
│   ├── real/    ← FFHQ images (70k available)
│   └── fake/    ← StyleGAN2/CelebA-HQ generated
└── test/
    ├── real/
    └── fake/
```

Download FFHQ: https://github.com/NVlabs/ffhq-dataset  
Download StyleGAN2 samples: https://github.com/NVlabs/stylegan2

### 2. Video Dataset (FaceForensics++ / DFDC)

```
datasets/videos/
├── train/
│   ├── real/    ← .mp4 original sequences
│   └── fake/    ← .mp4 manipulated (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
└── test/
    ├── real/
    └── fake/
```

Download FF++: https://github.com/ondyari/FaceForensics  
Download DFDC: https://ai.facebook.com/datasets/dfdc/

### 3. Audio Dataset (ASVspoof / FakeAVCeleb)

```
datasets/audio/
├── train/
│   ├── real/    ← .wav genuine speech
│   └── fake/    ← .wav spoofed/cloned speech
└── test/
    ├── real/
    └── fake/
```

Download ASVspoof 2019: https://www.asvspoof.org/  
Download FakeAVCeleb: https://github.com/DASH-Lab/FakeAVCeleb

### 4. News Dataset (LIAR / FakeNewsNet)

```
datasets/news/
├── train.json    ← [{"text": "...", "label": 0/1}, ...]
└── test.json
```

Convert LIAR dataset using `notebooks/1_dataset_collection.ipynb`.

Download LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

### 5. Document Dataset (MIDV-500 / RVL-CDIP)

```
datasets/documents/
├── train/
│   ├── real/    ← .jpg/.png genuine documents
│   └── fake/    ← .jpg/.png forged documents
└── test/
    ├── real/
    └── fake/
```

Optionally add `.txt` companion files with OCR text for each image.

---

## Training

### Train All Models

```bash
# Image model
python training/image_training.py

# Video model
python training/video_training.py

# Audio model
python training/audio_training.py

# News model
python training/news_training.py

# Document model
python training/document_training.py
```

### Incremental Training

Image and video models support incremental training — each run processes a fresh chunk of unseen data and resumes from the last checkpoint:

```bash
# Run multiple times to train on progressively more data
python training/image_training.py   # Run 1: images 0-2000
python training/image_training.py   # Run 2: images 2000-4000
python training/image_training.py   # Run 3: images 4000-6000
```

Progress is tracked in `datasets/image_train_progress.json`.

### Configuration

Edit `configs/model_configs.yaml` to adjust hyperparameters:

```yaml
image_model:
  input_size: 256
  batch_size: 32
  epochs: 5           # epochs per run (incremental)
  learning_rate: 0.0001

video_model:
  frame_size: 224
  sequence_length: 16
  batch_size: 4       # small due to memory
  epochs: 5
```

Edit `configs/training_configs.yaml` for general settings:

```yaml
general:
  seed: 42
  device: "cuda"      # or "cpu"
  early_stopping_patience: 5
```

---

## Checkpoints

Trained models are saved to `saved_models/`:

| File | Description |
|------|-------------|
| `image_model_best.pth` | Best image model by AUC |
| `image_model_latest.pth` | Latest image model checkpoint |
| `video_model_best.pth` | Best video model |
| `audio_model_best.pth` | Best audio model |
| `news_model_best.pth` | Best news model |
| `news_tokenizer.json` | Fitted news tokenizer |
| `document_model_best.pth` | Best document model |
| `document_tokenizer.json` | Document character tokenizer |

---

## Monitoring

Training logs are written to `logs/`:
- `logs/image_training.log`
- `logs/video_training.log`
- `logs/audio_training.log`
- `logs/news_training.log`
- `logs/document_training.log`

---

## Running Tests

```bash
pytest tests/ -v
```

Tests verify model forward passes, output shapes, and gradient flow without requiring trained weights.

---

## Minimum Dataset Sizes (Recommended)

| Modality | Train Real | Train Fake | Test Real | Test Fake |
|----------|-----------|-----------|----------|----------|
| Image | 5,000 | 5,000 | 1,000 | 1,000 |
| Video | 500 clips | 500 clips | 100 | 100 |
| Audio | 2,000 | 2,000 | 500 | 500 |
| News | 5,000 | 5,000 | 1,000 | 1,000 |
| Document | 1,000 | 1,000 | 200 | 200 |
