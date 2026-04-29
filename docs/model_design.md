# FakeNetBuster Model Design

## Overview

FakeNetBuster uses five specialized deep learning models, each trained from scratch for a specific modality.

---

## 1. Deepfake Image Model

**File:** `ml_models/deepfake_image/model.py`

**Architecture:** Triple-Branch CNN

```
Input (B, 3, 256, 256)
    ├── Spatial Branch (ResNet-style)
    │   └── Stem → ResBlock(32→64) → ResBlock(64→128) → ResBlock(128→256) → GAP → (B, 256)
    ├── Frequency Branch (High-pass + Laplacian)
    │   └── HP+Lap filters → Conv → ResBlock → GAP → FC → (B, 128)
    └── Noise Branch (SRM filter)
        └── SRM → Conv → ResBlock → GAP → FC → (B, 64)
            ↓
    Concat → (B, 448) → FC(256) → FC(64) → FC(1) → logit
```

**Key Design Choices:**
- SRM (Steganalysis Rich Model) filters detect copy-paste and generation artifacts
- Frequency branch captures GAN fingerprints invisible to spatial analysis
- Raw logit output — sigmoid applied at inference for calibrated probabilities
- GradCAM hooks on `spatial_branch.layer3` for explainability

**Training:** BCEWithLogitsLoss with pos_weight for class imbalance. Incremental chunked training.

---

## 2. Deepfake Video Model

**File:** `ml_models/deepfake_video/model.py`

**Architecture:** CNN + Bidirectional LSTM

```
Input (B, T, 3, 224, 224)
    ↓
FrameFeatureExtractor (per frame)
    4-block CNN → AdaptiveAvgPool(4×4) → FC(1024) → FC(512)
    → (B*T, 512) → reshape → (B, T, 512)
    ↓
TemporalConsistencyModule
    BiLSTM(512, hidden=512, layers=2) → (B, T, 1024)
    Attention: Linear(1024→128) → Tanh → Linear(128→1) → Softmax
    Context: weighted sum → (B, 1024)
    Classifier: FC(256) → FC(1) → Sigmoid
    ↓
Output: (B, 1) probability + (B, T) attention weights
```

**Key Design Choices:**
- Bidirectional LSTM captures both forward and backward temporal dependencies
- Attention mechanism identifies which frames are most suspicious
- Sliding window inference for long videos
- Frame-level attention weights used for timeline visualization

---

## 3. Deepfake Audio Model

**File:** `ml_models/deepfake_audio/model.py`

**Architecture:** CNN + Bidirectional GRU on Mel-Spectrogram

```
Input (B, 1, 128, T) — mel-spectrogram
    ↓
SpectrogramCNN
    4-block Conv2D (freq reduction, time preserved)
    AdaptiveAvgPool2d((4, None)) → freq_compress → (B, T, 256)
    ↓
TemporalGRU
    BiGRU(256, hidden=256, layers=2) → (B, T, 512)
    Attention: Linear(512→64) → Tanh → Linear(64→1) → Softmax
    Context: weighted sum → (B, 512)
    Classifier: FC(128) → FC(1) → Sigmoid
    ↓
Output: (B, 1) probability + (B, T) temporal attention
```

**Key Design Choices:**
- MaxPool only on frequency axis (not time) to preserve temporal resolution
- GRU preferred over LSTM for audio — faster convergence on sequential spectral data
- Pitch anomaly detection via librosa.pyin (post-processing, not model)
- Spectrogram visualization with anomalous region highlighting

---

## 4. Fake News Model

**File:** `ml_models/fake_news/model.py`

**Architecture:** Custom Transformer Encoder

```
Input: token_ids (B, seq_len=512)
    ↓
Embedding(vocab=30000, dim=256) + PositionalEncoding
    ↓
4× TransformerEncoderBlock
    MultiheadAttention(8 heads) + LayerNorm + Dropout
    FFN: Linear(256→512) → GELU → Linear(512→256) + LayerNorm
    ↓
AdaptiveAvgPool1d(1) → (B, 256)
    ↓
FC(128) → GELU → FC(1) → Sigmoid
    ↓
Output: (B, 1) probability + attention weights
```

**Key Design Choices:**
- Custom SimpleTokenizer (word-level) for multilingual support without external dependencies
- Sinusoidal positional encoding for sequence order awareness
- Attention weights from last transformer block used for suspicious token highlighting
- Heuristic analysis (clickbait patterns, caps ratio) runs in parallel with model

---

## 5. Fake Document Model

**File:** `ml_models/fake_documents/model.py`

**Architecture:** Dual-Branch CNN + NLP

```
Input: image (B, 3, 512, 512) + optional token_ids (B, 256)
    ├── DocumentCNN (visual branch)
    │   5-block Conv2D → AdaptiveAvgPool(4×4) → FC(1024) → FC(512)
    │   → (B, 512) visual features
    └── TextAuthenticityClassifier (text branch, optional)
        Embedding(10000, 128) → Conv1D(128→256) → AdaptiveMaxPool → FC
        → (B, 256) text features
            ↓
    If text available: Concat(512+256=768) → FC(512) → FC(128) → FC(1) → Sigmoid
    If visual only:    FC(512) → FC(128) → FC(1) → Sigmoid
```

**Key Design Choices:**
- Graceful degradation: works without OCR text (visual-only mode)
- Character-level DocumentTokenizer for robustness with OCR noise
- ELA (Error Level Analysis) runs as post-processing for copy-paste detection
- Circular Hough Transform for seal/stamp region detection

---

## Generation Method Detection

All models include heuristic classification of likely AI generation method:

| Model | Method Detection |
|-------|-----------------|
| Image | GAN fingerprints → "GAN-based face swap"; low noise → "Diffusion Model" |
| Video | High prob → "GAN face swap (FaceSwap/DeepFaceLab)"; medium → "Face reenactment" |
| Audio | High prob → "Neural voice cloning (VITS/YourTTS)"; medium → "TTS synthesis" |
| News | Always → "AI-generated or human-written misinformation" |
| Document | High prob → "Digital forgery (Photoshop/GIMP)"; medium → "Template generation" |
