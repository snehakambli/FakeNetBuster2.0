# FakeNetBuster System Architecture

## Overview

FakeNetBuster is a multi-modal fake content detection platform with five specialized AI models, a FastAPI backend, and a React frontend.

## Architecture Diagram

```
User Browser (React + TailwindCSS)
        │
        ▼
FastAPI Backend (port 8000)
        │
   ┌────┴────┐
   │         │
Upload    Analyze
Routes    Routes
   │         │
   └────┬────┘
        │
  Analysis Engine
        │
  Content Router
  (auto-detects type)
        │
   ┌────┼────┬────┬────┐
   │    │    │    │    │
Image Video Audio News Doc
Model Model Model Model Model
   │    │    │    │    │
   └────┼────┴────┴────┘
        │
 Explainability
 (GradCAM / ELA / Spectrogram)
        │
 Report Generator
        │
 MongoDB (reports storage)
```

## Components

### Frontend (React + Vite + TailwindCSS)
- Dark cybersecurity theme
- Drag-and-drop file upload
- Real-time analysis progress
- Interactive report viewer with charts
- Dashboard with history and statistics

### Backend (FastAPI)
- `/upload` - File upload with validation
- `/analyze/file` - File analysis endpoint
- `/analyze/news` - Text/URL news analysis
- `/report/{id}` - Report retrieval
- `/report/history` - Analysis history

### ML Models

| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| Image | Dual-branch CNN (Spatial + Frequency) | 256×256 RGB | Fake probability |
| Video | CNN + Bidirectional LSTM | T×224×224 frames | Fake prob + frame attention |
| Audio | CNN + Bidirectional GRU | Mel-spectrogram | Fake prob + time attention |
| News | Custom Transformer | Token sequence | Fake probability |
| Document | CNN + NLP classifier | 512×512 + OCR text | Fake probability |

### Explainability
- **Images**: GradCAM heatmaps on last conv layer
- **Videos**: Frame-level attention weights + suspicious frame extraction
- **Audio**: Temporal attention + spectrogram anomaly highlighting
- **Documents**: Error Level Analysis (ELA) + seal detection
- **News**: Token-level attention weights

## Data Flow

1. User uploads file via drag-and-drop
2. Backend saves file to `uploads/`
3. Content type auto-detected from extension
4. Routed to appropriate ML model
5. Inference runs with explainability
6. Report generated and saved to `reports/generated/`
7. Visualizations saved to `reports/visualizations/`
8. Report stored in MongoDB
9. Results returned to frontend

## Directory Structure

```
FakeNetBuster/
├── frontend/          # React application
├── backend/           # FastAPI server
├── ml_models/         # Model architectures + training + inference
├── inference/         # Inference pipeline + content router
├── training/          # Training entry points
├── reports/           # Report generator + templates
├── datasets/          # Training data (user-provided)
├── saved_models/      # Trained model checkpoints
├── configs/           # YAML configuration files
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit tests
├── utils/             # Shared utilities
└── docs/              # Documentation
```
