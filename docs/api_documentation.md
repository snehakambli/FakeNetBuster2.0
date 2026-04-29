# FakeNetBuster API Documentation

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### POST /upload/

Upload a file for analysis.

**Request:** `multipart/form-data`
- `file`: The file to upload (image, video, audio, or document)

**Response:**
```json
{
  "file_id": "abc123def456",
  "filename": "photo.jpg",
  "file_path": "uploads/abc123def456.jpg",
  "content_type_detected": "image",
  "file_size_mb": 1.24,
  "message": "File uploaded successfully. Use /analyze to process."
}
```

**Supported formats:**
- Images: `.jpg`, `.jpeg`, `.png`
- Videos: `.mp4`, `.mov`, `.avi`
- Audio: `.wav`, `.mp3`
- Documents: `.pdf`, `.jpg`, `.png`

---

### POST /analyze/file

Analyze an uploaded file.

**Request body:**
```json
{
  "file_path": "uploads/abc123def456.jpg",
  "content_type_hint": "image"
}
```

`content_type_hint` is optional. If omitted, content type is auto-detected.

**Response (image example):**
```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "content_type": "image",
  "prediction": "fake",
  "confidence_score": 0.93,
  "fake_probability": 0.93,
  "risk_level": "CRITICAL",
  "model_used": "DeepfakeImageModel (Dual-Branch CNN)",
  "possible_generation_method": "GAN-based face swap (color boundary mismatch)",
  "detected_anomalies": [
    "GAN frequency fingerprint detected",
    "color blending mismatch at face boundary",
    "high-confidence manipulation signature"
  ],
  "analysis_details": {
    "noise_inconsistency_score": 12.45,
    "color_inconsistency_score": 0.67,
    "gradcam_available": true,
    "visualization_path": "reports/visualizations/abc123_gradcam.jpg"
  },
  "processing_time_sec": 0.234,
  "summary": "Content classified as FAKE with 93.0% confidence. Risk level: CRITICAL. Key anomalies: GAN frequency fingerprint detected, color blending mismatch.",
  "report_path": "reports/generated/550e8400.json"
}
```

**Response (video example):**
```json
{
  "content_type": "video",
  "prediction": "fake",
  "confidence_score": 0.87,
  "analysis_details": {
    "suspicious_frames": [45, 78, 102, 156],
    "suspicious_timestamps_sec": [1.5, 2.6, 3.4, 5.2],
    "total_frames_analyzed": 48,
    "video_duration_sec": 10.5,
    "fps": 30.0,
    "manipulated_regions": ["face boundary mismatch"],
    "timeline_visualization_path": "reports/visualizations/video_suspicious_frames.jpg"
  }
}
```

---

### POST /analyze/news

Analyze news text or URL for fake news detection.

**Request body:**
```json
{
  "text": "Breaking news: Scientists discover miracle cure...",
  "url": null
}
```

Or with URL:
```json
{
  "text": null,
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "content_type": "news",
  "prediction": "fake",
  "confidence_score": 0.78,
  "analysis_details": {
    "word_count": 145,
    "suspicious_tokens": ["shocking", "miracle", "secret"],
    "text_features": {
      "clickbait_indicators": 3,
      "credibility_indicators": 0,
      "exclamation_ratio": 0.025,
      "caps_ratio": 0.12
    },
    "text_preview": "Breaking news: Scientists discover..."
  }
}
```

---

### POST /analyze/full

Unified endpoint accepting file path, text, or URL.

**Query parameters:**
- `file_path` (optional): Path to uploaded file
- `text` (optional): News text
- `url` (optional): News URL

---

### GET /report/{report_id}

Retrieve a specific analysis report.

**Response:** Full report JSON (same structure as analyze responses)

---

### GET /report/history

Get list of all analysis reports.

**Query parameters:**
- `limit` (optional, default=50): Maximum number of reports to return

**Response:**
```json
{
  "reports": [
    {
      "report_id": "550e8400-...",
      "timestamp": "2024-01-15T10:30:00Z",
      "content_type": "image",
      "prediction": "fake",
      "confidence_score": 0.93,
      "risk_level": "CRITICAL",
      "file_path": "uploads/abc123.jpg"
    }
  ],
  "total": 1
}
```

---

### DELETE /report/{report_id}

Delete a specific report.

**Response:**
```json
{"message": "Report 550e8400 deleted"}
```

---

## Risk Levels

| Level | Fake Probability | Description |
|-------|-----------------|-------------|
| CRITICAL | ≥ 0.85 | High confidence fake — immediate action recommended |
| HIGH | 0.65 – 0.84 | Likely fake — manual review recommended |
| MEDIUM | 0.45 – 0.64 | Uncertain — additional verification needed |
| LOW | < 0.45 | Likely real — low manipulation probability |

---

## Error Responses

```json
{
  "detail": "File not found"
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid file type, missing parameters) |
| 404 | File or report not found |
| 413 | File too large (max 500MB) |
| 500 | Internal server error (model not trained, inference failure) |

---

## Starting the Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API requires trained models in `saved_models/`. If models are not trained, inference endpoints will return an error with instructions to train first.
