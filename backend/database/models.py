"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime
import uuid


class AnalysisRequest(BaseModel):
    text_input: Optional[str] = None
    content_type_hint: Optional[str] = None  # override auto-detection


class AnalysisDetails(BaseModel):
    suspicious_frames: Optional[List[int]] = None
    suspicious_timestamps_sec: Optional[List[float]] = None
    total_frames_analyzed: Optional[int] = None
    video_duration_sec: Optional[float] = None
    fps: Optional[float] = None
    manipulated_regions: Optional[List[str]] = None
    noise_inconsistency_score: Optional[float] = None
    color_inconsistency_score: Optional[float] = None
    pitch_anomaly_score: Optional[float] = None
    spectral_inconsistency_score: Optional[float] = None
    anomalous_time_frames: Optional[List[int]] = None
    ela_score: Optional[float] = None
    font_inconsistency_score: Optional[float] = None
    tampered_regions: Optional[List[str]] = None
    source_url: Optional[str] = None
    word_count: Optional[int] = None
    suspicious_tokens: Optional[List[str]] = None
    text_preview: Optional[str] = None
    visualization_path: Optional[str] = None
    spectrogram_visualization_path: Optional[str] = None
    ela_visualization_path: Optional[str] = None
    timeline_visualization_path: Optional[str] = None


class AnalysisReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    content_type: str
    file_path: Optional[str] = None
    prediction: str
    confidence_score: float
    fake_probability: float
    risk_level: str
    model_used: str
    possible_generation_method: str
    detected_anomalies: List[str] = []
    analysis_details: Optional[Dict[str, Any]] = None
    visualizations: List[str] = []
    summary: str
    processing_time_sec: float = 0.0
    error: Optional[str] = None


class ReportSummary(BaseModel):
    report_id: str
    timestamp: str
    content_type: str
    prediction: str
    confidence_score: float
    risk_level: str
    file_path: Optional[str] = None


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_path: str
    content_type_detected: str
    file_size_mb: float
    message: str
