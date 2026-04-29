# FakeNetBuster 2.0
## An AI-Powered Multi-Modal Fake Content Detection Platform

---

**Project Title:** FakeNetBuster 2.0: An AI-Powered Multi-Modal Fake Content Detection Platform  
**Version:** 2.0.0  
**Date:** March 2026  
**Domain:** Artificial Intelligence · Computer Vision · Natural Language Processing · Cybersecurity  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Literature Review](#5-literature-review)
6. [System Architecture](#6-system-architecture)
7. [Hardware and Software Requirements](#7-hardware-and-software-requirements)
8. [Dataset Description](#8-dataset-description)
9. [Proposed Algorithms and Model Design](#9-proposed-algorithms-and-model-design)
10. [Training Methodology](#10-training-methodology)
11. [Inference Pipeline](#11-inference-pipeline)
12. [Explainability and Visualization](#12-explainability-and-visualization)
13. [News Fact-Checking Engine](#13-news-fact-checking-engine)
14. [Backend API Implementation](#14-backend-api-implementation)
15. [Frontend Implementation](#15-frontend-implementation)
16. [Results and Evaluation](#16-results-and-evaluation)
17. [Challenges and Solutions](#17-challenges-and-solutions)
18. [Conclusion and Future Work](#18-conclusion-and-future-work)
19. [References](#19-references)

---

## 1. Abstract

FakeNetBuster 2.0 is a full-stack, AI-powered platform engineered to detect fake and manipulated digital content across five distinct modalities: images, videos, audio recordings, news articles, and identity documents. The system employs five independently designed and trained deep learning models, each architecturally optimized for its respective modality, and integrates them into a unified multimodal inference pipeline with automatic content-type routing.

The platform is built on a three-tier architecture comprising a React 18 frontend, a FastAPI asynchronous backend, and a PyTorch-based machine learning processing layer. Explainability is a first-class concern — the system generates GradCAM heatmaps for images, frame-level attention timelines for videos, mel-spectrogram anomaly visualizations for audio, and Error Level Analysis maps for documents. For news verification, the system integrates five external APIs: Google Fact Check Tools, ClaimBuster, NewsAPI, Google Custom Search, and Groq/Gemini large language models, producing an aggregated verdict with cross-referenced source links.

All analysis results are persisted as structured JSON reports in MongoDB, accessible through a RESTful API. The system is designed to run on consumer-grade hardware with a minimum of 4 GB GPU VRAM, making it accessible for deployment in newsrooms, research institutions, and content moderation pipelines.

---

## 2. Introduction

The proliferation of generative artificial intelligence has fundamentally altered the landscape of digital trust. Technologies such as Generative Adversarial Networks (GANs), diffusion models, neural voice synthesis, and large language models have made it trivially easy to produce synthetic media that is indistinguishable from authentic content to the human eye and ear. The consequences are far-reaching: fabricated political speeches, cloned voices used in financial fraud, forged identity documents, and algorithmically generated misinformation campaigns have all emerged as documented threats to democratic institutions, public safety, and individual privacy.

Existing detection tools suffer from three critical limitations. First, they are predominantly single-modality — a tool that detects deepfake faces cannot analyze audio or text. Second, they operate as black boxes, providing a verdict without any explanation of the evidence that led to it. Third, they typically require cloud API access, creating latency, cost, and data privacy concerns that make them unsuitable for sensitive or high-volume use cases.

FakeNetBuster addresses all three limitations. It provides a single, unified platform that covers all major fake content modalities, explains every decision through visual and textual evidence, and runs entirely on local hardware without dependency on external ML inference APIs. The system is designed for journalists verifying breaking news, security analysts investigating fraud, content moderators screening uploads, and researchers studying the spread of synthetic media.

---

## 3. Problem Statement

Digital misinformation does not manifest in isolation. A coordinated disinformation campaign may simultaneously deploy a deepfake video, a voice-cloned audio clip, a forged document, and a fabricated news article — all referencing the same false narrative. Addressing this threat requires a system capable of analyzing all these modalities in a unified, consistent, and explainable manner.

The specific problems this project addresses are:

1. **Fragmentation of detection tools** — No single open-source platform covers image, video, audio, document, and news fake detection simultaneously.

2. **Lack of explainability** — Existing tools provide a binary verdict without indicating which regions, frames, or tokens triggered the detection, making results difficult to trust or act upon.

3. **Dependency on external APIs** — Cloud-based detection services introduce latency, recurring costs, and data privacy risks that are unacceptable in many deployment contexts.

4. **Static fact-checking** — Rule-based fake news detectors rely on fixed keyword lists and cannot adapt to novel misinformation patterns or cross-reference live news databases.

5. **No unified report structure** — Without a standardized report format, integrating detection results into downstream workflows (legal review, editorial processes, moderation queues) is difficult.

---

## 4. Objectives

The primary objectives of FakeNetBuster 2.0 are as follows:

- Design and implement five specialized deep learning models for image, video, audio, news, and document fake detection, each trained from scratch on publicly available datasets.
- Build a multimodal inference pipeline that automatically detects content type and routes inputs to the appropriate model without user intervention.
- Provide explainability through GradCAM heatmaps, temporal attention visualization, mel-spectrogram anomaly highlighting, and Error Level Analysis.
- Integrate a multi-signal news fact-checking engine combining heuristic analysis, Google Fact Check API, ClaimBuster, NewsAPI cross-referencing, and LLM-based reasoning (Groq/Gemini).
- Expose all functionality through a RESTful FastAPI backend with asynchronous processing.
- Build an intuitive React frontend supporting drag-and-drop file upload, real-time analysis feedback, and interactive report viewing.
- Persist all analysis reports in MongoDB with full retrieval, history, and deletion support.
- Support incremental model training with checkpoint resumption, early stopping, and mixed-precision training for efficient use of limited GPU resources.
- Design the system to run on consumer-grade hardware (minimum 4 GB VRAM) without requiring cloud ML inference APIs.

---

## 5. Literature Review

### 5.1 Deepfake Image Detection

Early deepfake detection relied on hand-crafted features such as eye blinking patterns and facial landmark inconsistencies. Rossler et al. (2019) introduced FaceForensics++, a large-scale benchmark dataset and baseline CNN detector that established the standard for video and image deepfake evaluation. Subsequent work by Wang et al. (2020) demonstrated that GAN-generated images leave frequency-domain fingerprints detectable by high-pass filtering, motivating the frequency branch in FakeNetBuster's image model. The Steganalysis Rich Model (SRM) filter, originally developed for image steganography detection by Fridrich and Kodovsky (2012), was later adapted for deepfake detection by Zhou et al. (2017), forming the basis of the noise residual branch.

### 5.2 Deepfake Video Detection

Temporal inconsistency detection was formalized by Sabir et al. (2019), who demonstrated that recurrent neural networks applied to frame sequences outperform single-frame CNN detectors on compressed video. The attention mechanism for identifying suspicious frames was introduced in the context of video deepfake detection by Zheng et al. (2021), who showed that attention-weighted temporal pooling significantly improves detection of partial face swaps.

### 5.3 Audio Deepfake Detection

The ASVspoof challenge series (2015–2021) established the benchmark datasets and evaluation protocols for automatic speaker verification spoofing detection. Mel-spectrogram CNN approaches were shown to be competitive with hand-crafted feature methods by Lavrentyeva et al. (2017). The combination of CNN feature extraction with recurrent temporal modeling was demonstrated by Tak et al. (2021) to achieve state-of-the-art performance on the ASVspoof 2019 dataset.

### 5.4 Fake News Detection

The LIAR dataset (Wang, 2017) provided the first large-scale benchmark for fake news detection with fine-grained veracity labels. Transformer-based approaches, following the success of BERT (Devlin et al., 2018), have dominated the field since 2019. FakeNetBuster implements a custom Transformer encoder trained from scratch, avoiding dependency on pre-trained language models while retaining the architectural advantages of self-attention.

### 5.5 Document Forgery Detection

Error Level Analysis (ELA), introduced by Krawetz (2007), remains a widely used technique for detecting JPEG re-compression artifacts introduced by copy-paste forgery. Deep learning approaches combining CNN visual analysis with OCR-based text verification have been explored by Jain et al. (2019) for identity document authentication.

---

## 6. System Architecture

### 6.1 Architectural Overview

FakeNetBuster follows a three-tier architecture with clear separation of concerns:

- **Presentation Layer** — React 18 frontend (Vite + Tailwind CSS)
- **Application Layer** — FastAPI asynchronous backend (Python 3.12)
- **Processing Layer** — PyTorch ML modules + external API integrations

```
┌─────────────────────────────────────────────────────────────────┐
│                    React 18 Frontend (Vite)                      │
│   File Upload │ Analysis Results │ Report History │ Dashboard    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST (Axios)
┌────────────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend (Async, Port 8000)              │
│   /upload  │  /analyze/file  │  /analyze/news  │  /report       │
│   /preview/video  (ffmpeg transcoding)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               Multimodal Inference Pipeline                       │
│                                                                   │
│   Content Router → [Image | Video | Audio | Document | News]     │
│                              │                                    │
│          ┌───────────────────┼──────────────────┐                │
│          ▼                   ▼                  ▼                │
│    PyTorch Models     Explainability      Fact-Check APIs         │
│    (5 modalities)     (GradCAM/ELA/       (Google/NewsAPI/        │
│                        Spectrogram)        Groq/Gemini)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              Report Generator + MongoDB Persistence              │
│         Structured JSON Reports │ Risk Classification             │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Directory Structure

```
FakeNetBuster/
├── backend/
│   ├── main.py                    FastAPI application entry point
│   ├── routes/
│   │   ├── upload_routes.py       POST /upload/
│   │   ├── analysis_routes.py     POST /analyze/file, /analyze/news
│   │   ├── report_routes.py       GET/DELETE /report/
│   │   └── preview_routes.py      GET /preview/video (ffmpeg transcoding)
│   ├── services/
│   │   ├── analysis_engine.py     Bridges API to inference pipeline
│   │   ├── file_handler.py        File upload and validation
│   │   ├── report_service.py      Report retrieval and history
│   │   ├── news_analyzer.py       News analysis orchestrator
│   │   └── news_fact_checker.py   Multi-API fact-checking service
│   └── database/
│       ├── mongo_client.py        Async MongoDB client (Motor)
│       └── models.py              Pydantic request/response models
├── ml_models/
│   ├── deepfake_image/            EfficientNet-B4 + Frequency + Noise branches
│   ├── deepfake_video/            EfficientNet-B3 + BiLSTM + Attention
│   ├── deepfake_audio/            CNN + BiGRU on Mel-Spectrogram
│   ├── fake_news/                 Custom Transformer Encoder
│   └── fake_documents/            Dual-Branch CNN + NLP
├── inference/
│   ├── multimodal_inference.py    Unified pipeline entry point
│   ├── content_router.py          Content type detection and routing
│   ├── audio_inference.py         Audio inference + spectrogram viz
│   ├── video_inference.py         Video inference + frame timeline
│   ├── image_inference.py         Image inference + GradCAM
│   ├── document_inference.py      Document inference + ELA
│   └── news_inference.py          News inference wrapper
├── reports/
│   ├── report_generator.py        Structured JSON report generation
│   ├── templates/                 Per-modality report templates
│   └── generated/                 Saved analysis reports (UUID.json)
├── training/                      Shared training utilities
├── datasets/                      Training and test data
├── saved_models/                  Trained model checkpoints (.pth)
├── configs/
│   ├── system_configs.yaml        Server, database, storage, API keys
│   ├── model_configs.yaml         Per-model hyperparameters
│   └── training_configs.yaml      Augmentation, optimizer, scheduler
├── docs/                          Documentation
└── frontend/                      React 18 + Vite + Tailwind CSS
```

### 6.3 Data Flow

```
1. User uploads file or enters text/URL via frontend
2. Backend saves file to uploads/ with UUID filename
3. Content type auto-detected from file extension and MIME type
4. Input routed to appropriate ML inference module
5. Model performs inference; explainability components generate evidence
6. Results aggregated into structured report with risk classification
7. Report saved as UUID.json in reports/generated/
8. Report stored in MongoDB for history and retrieval
9. Complete report returned to frontend via JSON response
10. Frontend renders results with visualizations and confidence indicators
```

### 6.4 Risk Classification

All analysis results are classified into four risk levels based on fake probability:

| Risk Level | Fake Probability Range | Interpretation |
|------------|----------------------|----------------|
| LOW | 0.00 – 0.44 | Content appears authentic |
| MEDIUM | 0.45 – 0.64 | Suspicious — manual review recommended |
| HIGH | 0.65 – 0.84 | Likely manipulated — strong evidence of fakery |
| CRITICAL | 0.85 – 1.00 | Almost certainly fake — high-confidence manipulation |

---

## 7. Hardware and Software Requirements

### 7.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 4 GB VRAM (RTX 2050) | NVIDIA 8 GB+ VRAM (RTX 3060+) |
| CPU | 4-core, 2.5 GHz | 8-core, 3.5 GHz |
| RAM | 8 GB | 16 GB |
| Storage | 50 GB SSD | 200 GB+ SSD |
| CUDA | 11.8+ | 12.x |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

All batch sizes in `configs/model_configs.yaml` are tuned for 4 GB VRAM with mixed-precision training (FP16). CPU-only operation is supported but significantly slower for image and video models.

### 7.2 Software Stack

**Backend**

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12 | Runtime |
| FastAPI | ≥ 0.109 | Async REST framework |
| Uvicorn | ≥ 0.27 | ASGI server |
| Motor | ≥ 3.3 | Async MongoDB driver |
| PyMongo | ≥ 4.6 | MongoDB client |
| Pydantic | ≥ 2.5 | Data validation |
| PyYAML | ≥ 6.0 | Configuration loading |
| imageio-ffmpeg | ≥ 0.6 | Bundled ffmpeg for video transcoding |

**Machine Learning**

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning framework |
| torchvision | ≥ 0.15 | EfficientNet backbones, transforms |
| torchaudio | ≥ 2.0 | Audio processing |
| NumPy | ≥ 1.24 | Numerical computing |
| scikit-learn | ≥ 1.3 | Metrics (AUC, F1, accuracy) |
| librosa | ≥ 0.10 | Audio feature extraction |
| OpenCV | ≥ 4.8 | Video frame extraction |
| Pillow | ≥ 10.0 | Image loading and processing |
| pytesseract | ≥ 0.3 | OCR for document text extraction |

**Frontend**

| Package | Version | Purpose |
|---------|---------|---------|
| React | 18.2 | UI framework |
| Vite | Latest | Build tool |
| Tailwind CSS | Latest | Utility-first styling |
| Axios | Latest | HTTP client |
| React Dropzone | Latest | File upload UI |
| Lucide React | Latest | Icon library |
| React Hot Toast | Latest | Notifications |

**Database**

| Component | Version | Purpose |
|-----------|---------|---------|
| MongoDB | ≥ 4.4 | Report and upload metadata storage |

---

## 8. Dataset Description

### 8.1 Image Dataset

- **Real images:** FFHQ (Flickr-Faces-HQ) — 70,000 high-quality face photographs at 1024×1024 resolution, collected from Flickr with diverse demographics, ages, and lighting conditions.
- **Fake images:** StyleGAN2-generated faces and CelebA-HQ GAN outputs.
- **Current split:** ~33,437 fake training images, ~5,207 real training images; ~5,246 fake test, ~5,207 real test.
- **Format:** JPEG, stored in `datasets/images/train/{real,fake}/` and `datasets/images/test/{real,fake}/`.
- **Preprocessing:** Resize to 380×380 (EfficientNet-B4 native resolution), normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

### 8.2 Video Dataset

- **Real:** Original video sequences from FaceForensics++ (FF++) and DFDC (DeepFake Detection Challenge).
- **Fake:** Manipulated videos using Deepfakes, Face2Face, FaceSwap, and NeuralTextures methods from the FF++ dataset.
- **Current split:** 83 training clips per class (real/fake), 83 test clips per class.
- **Format:** MP4, stored in `datasets/videos/train/{real,fake}/`.
- **Preprocessing:** Extract frames at every 5th frame using OpenCV, resize to 224×224, normalize with ImageNet statistics.

### 8.3 Audio Dataset

- **Source:** ASVspoof 2017 V2 dataset — a benchmark for automatic speaker verification spoofing detection.
- **Real:** Genuine speech recordings (labeled "genuine" in protocol files).
- **Fake:** Spoofed/voice-cloned speech (labeled "spoof" in protocol files).
- **Current split:** 1,017 training files per class; development set used for testing.
- **Format:** WAV (16 kHz, mono), stored in `datasets/audio/train/{real,fake}/`.
- **Preprocessing:** Load at 16 kHz, compute mel-spectrogram (n_mels=128, n_fft=1024, hop_length=512), convert to dB scale, normalize per-sample, pad/truncate to 128 frames.

### 8.4 News Dataset

- **Source:** LIAR dataset — 12,836 labeled statements from PolitiFact with six veracity levels.
- **Label mapping:** pants-fire / false / barely-true → fake (1); true / mostly-true → real (0).
- **Format:** JSON `[{"text": "...", "label": 0/1}]`, stored in `datasets/news/train.json` and `test.json`.
- **Tokenization:** Custom word-level tokenizer (SimpleTokenizer) with vocabulary of 30,000 words, max sequence length 512 tokens.

### 8.5 Document Dataset

- **Real:** Genuine identity documents from MIDV-500 dataset (passports, ID cards, driver's licenses).
- **Fake:** Forged documents from RVL-CDIP and custom augmented forgeries.
- **Format:** JPEG/PNG at 512×512, optionally with companion `.txt` OCR files.
- **Stored in:** `datasets/documents/train/{real,fake}/`.

---

## 9. Proposed Algorithms and Model Design

### 9.1 Deepfake Image Detection — Triple-Branch CNN

#### Architecture Rationale

GAN-generated images leave artifacts simultaneously in three domains: spatial blending errors visible to the eye, frequency-domain fingerprints from the GAN's upsampling operations, and noise residual inconsistencies from the generation process. A single-branch CNN captures only spatial artifacts. The triple-branch architecture ensures all three artifact types are detected independently and fused.

#### Architecture Specification

```
Input: (B, 3, 380, 380)
        │
        ├── Spatial Branch (EfficientNet-B4 backbone)
        │   features → AdaptiveAvgPool2d(1) → Flatten → (B, 1792)
        │
        ├── Frequency Branch
        │   Gray → High-pass kernel [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        │        + Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]
        │   → Conv2d(2→32) → BN → ReLU → Conv2d(32→64, stride=2)
        │   → BN → ReLU → AdaptiveAvgPool2d(1) → FC(64→128) → (B, 128)
        │
        └── Noise Branch (SRM filter)
            SRM 5×5 kernel (per channel) → Conv2d(3→32) → BN → ReLU
            → Conv2d(32→64, stride=2) → BN → ReLU
            → AdaptiveAvgPool2d(1) → FC(64→64) → (B, 64)
                    │
            Concatenate: (B, 1984)
                    │
            Classifier:
            FC(1984→512) → ReLU → Dropout(0.4)
            → FC(512→128) → ReLU → Dropout(0.2)
            → FC(128→1) → raw logit
                    │
Output: (B, 1) logit — sigmoid applied at inference
```

#### Key Design Decisions

- **EfficientNet-B4 backbone** provides 1,792-dimensional spatial features with compound scaling across depth, width, and resolution. Pre-trained on ImageNet-1K.
- **High-pass filter** `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` and **Laplacian filter** `[[0,1,0],[1,-4,1],[0,1,0]]` detect GAN frequency fingerprints that are invisible to spatial analysis.
- **SRM (Steganalysis Rich Model) filter** — a 5×5 noise residual kernel applied per channel — detects copy-paste and generation noise patterns originally developed for image steganography detection.
- **Raw logit output** with BCEWithLogitsLoss is numerically more stable than sigmoid + BCELoss.
- **Training schedule:** Backbone frozen for first 3 epochs, then unfrozen with differential learning rate (backbone: lr × 0.1, other branches: lr × 1.0).

#### Post-Processing Analysis

- **GradCAM heatmap:** Computed on the last EfficientNet-B4 feature block, overlaid on the original image to highlight suspicious regions.
- **Noise inconsistency score:** SRM residual variance ratio between center and border regions.
- **Color inconsistency score:** HSV saturation variance between face center and boundary regions.

---

### 9.2 Deepfake Video Detection — EfficientNet-B3 + BiLSTM

#### Architecture Rationale

Video deepfakes are temporal phenomena. A single frame may appear authentic, but inconsistencies emerge when comparing frames across time — particularly in face boundary regions, eye blink patterns, and lighting transitions. The model processes sequences of frames through a CNN backbone for per-frame feature extraction, then models temporal dependencies with a bidirectional LSTM and attention mechanism.

#### Architecture Specification

```
Input: (B, T=16, 3, 224, 224)
        │
        Reshape: (B×T, 3, 224, 224)
        │
EfficientNet-B3 Frame Feature Extractor:
  features → AdaptiveAvgPool2d(1) → Flatten → (B×T, 1536)
  Reshape: (B, T, 1536)
        │
Bidirectional LSTM:
  BiLSTM(input=1536, hidden=256, layers=2, bidirectional=True)
  → (B, T, 512)
        │
Temporal Attention:
  Linear(512→64) → Tanh → Linear(64→1) → Softmax(dim=T)
  → attention weights (B, T, 1)
  Context vector: weighted sum → (B, 512)
        │
Classifier:
  FC(512→128) → ReLU → Dropout(0.4) → FC(128→1) → raw logit
        │
Output: (B, 1) logit + (B, T) per-frame attention weights
```

#### Inference Algorithm

1. Extract frames from video using OpenCV at every 5th frame (sample_rate=5).
2. Resize each frame to 224×224, normalize with ImageNet statistics.
3. Process frames in sliding windows of size T=16 with step=8 (50% overlap).
4. For each window: EfficientNet-B3 extracts per-frame features → BiLSTM models temporal context → attention identifies suspicious frames.
5. Apply Test-Time Augmentation (TTA): average probabilities from normal and horizontally flipped inputs.
6. Aggregate window probabilities: `overall_prob = 0.6 × mean_prob + 0.4 × max_prob`.
7. Identify suspicious frames: windows with probability > 70th percentile, top-3 attention frames per window.

#### Why Bidirectional LSTM

A bidirectional LSTM processes the frame sequence in both forward and backward directions, capturing both "what came before" and "what comes after" each frame. This is critical for detecting face swap artifacts that may only be visible when comparing a frame to its temporal neighbors in both directions.

---

### 9.3 Deepfake Audio Detection — CNN + BiGRU on Mel-Spectrogram

#### Architecture Rationale

Voice cloning and text-to-speech synthesis leave artifacts in the mel-spectrogram — particularly in high-frequency bands and temporal transitions between phonemes. The model converts raw audio to a mel-spectrogram and processes it with a CNN for frequency-domain feature extraction, followed by a BiGRU for temporal modeling.

#### Architecture Specification

```
Input: Raw audio → Mel-spectrogram (B, 1, 128, 128)
        │
SpectrogramCNN:
  Block 1: Conv2d(1→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU
           → MaxPool2d(2×1)  [frequency axis only]
  Block 2: Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU
           → MaxPool2d(2×1)
  Block 3: Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU
           → MaxPool2d(2×1)
  Block 4: Conv2d(128→256) → BN → ReLU → AdaptiveAvgPool2d((4, None))
  Reshape: (B, T, 256×4=1024)
  freq_compress: Linear(1024→256) → ReLU → (B, T, 256)
        │
Bidirectional GRU:
  BiGRU(input=256, hidden=256, layers=2, bidirectional=True)
  → (B, T, 512)
        │
Temporal Attention:
  Linear(512→64) → Tanh → Linear(64→1) → Softmax(dim=T)
  Context vector: weighted sum → (B, 512)
        │
Classifier:
  FC(512→128) → ReLU → Dropout(0.3) → FC(128→1) → raw logit
        │
Output: (B, 1) logit + (B, T) temporal attention weights
```

#### Critical Design Decision: Frequency-Only Pooling

MaxPool is applied only on the frequency axis (MaxPool(2×1)), preserving the full temporal resolution of the spectrogram. This is essential because voice cloning artifacts often manifest as brief temporal glitches that would be destroyed by temporal pooling.

#### Confidence Calibration

Raw model probability is calibrated using a soft sigmoid stretching function to produce more meaningful confidence scores:

```
calibrated = 0.5 + 0.5 × (x / (1.0 + |x|))
where x = (raw_confidence - 0.5) × 6.0
```

This maps a raw confidence of 0.58 to approximately 0.70, and 0.70 to approximately 0.83, without artificially inflating borderline predictions.

#### Heuristic Fusion

Final probability is a weighted fusion of model output and acoustic heuristics:

```
fused_prob = 0.80 × model_prob + 0.20 × heuristic_prob
heuristic_prob = 0.5 × pitch_anomaly_score + 0.5 × spectral_inconsistency_score
```

- **Pitch anomaly detection:** `librosa.pyin()` extracts fundamental frequency (F0). Sudden F0 jumps beyond mean + 2×std are flagged. Score = anomaly_count / total_voiced_frames.
- **Spectral inconsistency:** Variance ratio of high-frequency (mel bins 64–128) vs. low-frequency (mel bins 0–63) bands. Voice cloning produces unnatural high-frequency energy patterns.

---

### 9.4 Fake News Detection — Custom Transformer Encoder

#### Architecture Rationale

The fake news model is a custom Transformer encoder built entirely from PyTorch primitives — no external NLP libraries, no pre-trained BERT or HuggingFace models. This makes the system fully self-contained and deployable without internet access. A custom word-level tokenizer is trained on the dataset vocabulary, providing multilingual support without language-specific preprocessing.

#### Architecture Specification

```
Input: token_ids (B, seq_len=512)
        │
Embedding: nn.Embedding(vocab_size=30,000, embed_dim=256)
        │
Sinusoidal Positional Encoding:
  PE[pos, 2i]   = sin(pos / 10000^(2i/256))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/256))
  → (B, seq_len, 256)
        │
4× TransformerEncoderBlock:
  MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
  → Add & LayerNorm
  FFN: Linear(256→512) → GELU → Dropout(0.1) → Linear(512→256)
  → Add & LayerNorm
  → (B, seq_len, 256) + attention_weights (B, 8, seq_len, seq_len)
        │
AdaptiveAvgPool1d(1) → (B, 256)
        │
Classifier:
  FC(256→128) → GELU → Dropout(0.1) → FC(128→1) → Sigmoid
        │
Output: (B, 1) fake probability + last-layer attention weights
```

#### SimpleTokenizer

A custom word-level tokenizer is built from training data:
- Special tokens: `<PAD>=0`, `<UNK>=1`, `<CLS>=2`, `<SEP>=3`
- Vocabulary: top 30,000 words by frequency from training corpus
- Encoding: `[<CLS>] + words + [<SEP>]`, padded/truncated to max_len=512
- Saved as JSON for deterministic reuse at inference time

---

### 9.5 Fake Document Detection — Dual-Branch CNN + NLP

#### Architecture Rationale

Document forgery requires both visual analysis (pixel-level tampering, layout anomalies, seal detection) and text analysis (OCR-extracted content authenticity). The dual-branch architecture handles both simultaneously, with graceful degradation to visual-only mode when OCR is unavailable.

#### Architecture Specification

```
Input: image (B, 3, 512, 512) + optional token_ids (B, 256)
        │
DocumentCNN (Visual Branch):
  5-block Conv2D with BatchNorm and ReLU
  Block 1–4: MaxPool2d(2×2) after each block
  Block 5: AdaptiveAvgPool2d(4×4)
  FC: Linear(512×16→1024) → ReLU → Dropout(0.4) → Linear(1024→512) → (B, 512)
        │
TextAuthenticityClassifier (Text Branch, optional):
  Embedding(vocab=10,000, dim=128) → (B, L, 128)
  Conv1d(128→128, k=3) → ReLU → Conv1d(128→256, k=3) → ReLU
  AdaptiveMaxPool1d(1) → FC(256→256) → ReLU → (B, 256)
        │
Fusion:
  If text available: Concat(512+256=768) → FC(768→512) → ReLU
                     → Dropout(0.4) → FC(512→128) → ReLU
                     → Dropout(0.2) → FC(128→1) → Sigmoid
  If visual only:    FC(512→128) → ReLU → Dropout(0.3) → FC(128→1) → Sigmoid
        │
Output: (B, 1) fake probability
```

#### DocumentTokenizer

Character-level tokenization for robustness with OCR noise:
- 44-character alphabet: a–z, 0–9, space, and common punctuation (.,/-:)
- Character-level encoding is more robust than word-level when OCR introduces typographic errors
- Max length: 256 characters

#### Post-Processing Analysis

- **Error Level Analysis (ELA):** Save image at JPEG quality=90, compute pixel-wise absolute difference between original and compressed version. High ELA score (>15.0) indicates copy-paste regions re-saved at different compression levels.
- **Seal detection:** Circular Hough Transform applied to detect circular stamp/seal regions. Absence of expected seals in official documents is flagged as an anomaly.

---
