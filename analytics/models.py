"""Pydantic schemas and dataclasses for the post-session analytics engine and final reports."""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class TelemetryPoint(BaseModel):
    """A single time-indexed snapshot of physical and speech performance."""
    timestamp: float = Field(..., description="Seconds since session start")
    shoulder_tilt_degrees: float = Field(default=0.0)
    spine_deviation_degrees: float = Field(default=0.0)
    hand_movement_velocity: float = Field(default=0.0)
    stability_index: float = Field(default=100.0)
    wpm: Optional[float] = Field(default=None)
    is_slumping: bool = Field(default=False)
    are_shoulders_uneven: bool = Field(default=False)
    is_fidgeting: bool = Field(default=False)


class SessionSummary(BaseModel):
    """High-level executive metrics for the presentation."""
    overall_score: float = Field(..., ge=0.0, le=100.0)
    duration_seconds: float = Field(..., ge=0.0)
    total_words_spoken: int = Field(default=0, ge=0)
    average_cadence_wpm: float = Field(default=0.0, ge=0.0)
    filler_word_count: int = Field(default=0, ge=0)
    filler_word_ratio_pct: float = Field(default=0.0, ge=0.0)
    posture_stability_score: float = Field(default=0.0, ge=0.0, le=100.0)


class RadarDimensions(BaseModel):
    """Normalized multi-dimensional skill ratings on a 0-100 scale."""
    posture: float = Field(..., ge=0.0, le=100.0)
    pacing: float = Field(..., ge=0.0, le=100.0)
    gesture_expressiveness: float = Field(..., ge=0.0, le=100.0)
    verbal_clarity: float = Field(..., ge=0.0, le=100.0)
    energy_consistency: float = Field(..., ge=0.0, le=100.0)


class TimeSeriesMetrics(BaseModel):
    """Temporal time-series arrays for presentation charting."""
    timestamps: List[float] = Field(default_factory=list)
    wpm_rolling: List[float] = Field(default_factory=list)
    posture_stability_rolling: List[float] = Field(default_factory=list)
    hand_fidget_energy: List[float] = Field(default_factory=list)


class DetectedDefect(BaseModel):
    """Anomalies and posture/speech defects flagged during the presentation."""
    defect_type: str
    timestamp_range: Tuple[float, float]
    severity: str = Field(description="LOW, MODERATE, HIGH")
    metric_value: str


class CoachingRecommendation(BaseModel):
    """Diagnostic recommendation retrieved from the offline RAG corpus."""
    source_transcript_id: str
    similarity_score: float
    topic: str
    recommendation: str


class PostSessionReport(BaseModel):
    """Final comprehensive evaluation report synthesized by the analytics engine."""
    session_id: str
    status: str = "COMPLETED"
    summary: SessionSummary
    radar_dimensions: RadarDimensions
    time_series_metrics: TimeSeriesMetrics
    detected_defects: List[DetectedDefect] = Field(default_factory=list)
    rag_coaching_recommendations: List[CoachingRecommendation] = Field(default_factory=list)
