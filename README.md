# FakeNetBuster 2.0

An AI-powered multi-modal fake content detection platform. Detects deepfakes and misinformation across **5 modalities** — images, videos, audio, news articles, and identity documents — running entirely on local hardware.

## Features

- **Image Detection** — GAN fingerprints, frequency artifacts, GradCAM heatmaps (CNN + Frequency branches)
- **Video Detection** — Temporal inconsistency, suspicious frame timestamps (EfficientNet + BiLSTM)
- **Audio Detection** — Voice cloning artifacts, pitch anomalies, mel-spectrogram analysis (CNN + BiGRU)
- **News Detection** — Clickbait patterns, fact-checking via Google, ClaimBuster, NewsAPI, Groq/Gemini (Transformer)
- **Document Detection** — ELA forgery analysis, font inconsistencies, OCR verification (CNN + NLP)

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React 18 + Vite + Tailwind CSS |
| Backend | FastAPI + Uvicorn |
| ML | PyTorch 2.0 + TorchVision + LibROSA + OpenCV |
| Database | MongoDB (Motor async driver) |

## Setup

### 1. Clone & configure

```bash
git clone https://github.com/your-username/FakeNetBuster.git
cd FakeNetBuster

# Copy and fill in your API keys
cp configs/system_configs.yaml.example configs/system_configs.yaml
```

### 2. Backend

```bash
pip install -r requirements.txt
python -m backend.main
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. MongoDB

Make sure MongoDB is running locally on port `27017`, or update the URI in `configs/system_configs.yaml`.

## API Keys Required (for News Detection)

Get free keys from:
- [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api)
- [NewsAPI](https://newsapi.org/)
- [Groq](https://console.groq.com/) or [Gemini](https://aistudio.google.com/)
- [Tavily](https://tavily.com/)

Add them to `configs/system_configs.yaml` (never commit this file).

## Project Structure

```
FakeNetBuster/
├── backend/          # FastAPI app, routes, services
├── frontend/         # React + Vite UI
├── ml_models/        # Model architectures + training scripts
├── inference/        # Inference pipeline per modality
├── datasets/         # Dataset arrangement scripts
├── configs/          # YAML configuration
├── docs/             # Architecture docs, project report
├── training/         # Training utilities
└── utils/            # Shared utilities
```

## Training Models

Each model has its own training script under `ml_models/<modality>/train.py`. Trained weights go in `saved_models/` (gitignored).

## License

MIT
