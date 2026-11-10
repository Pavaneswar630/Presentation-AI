"""System configuration and SLA performance thresholds for Presentation AI."""

from pydantic_settings import BaseSettings
from pydantic import Field


class PipelineSettings(BaseSettings):
    """Pipeline performance and SLA parameters."""
    
    # Target frame rates and SLAs
    target_fps: float = Field(default=30.0, description="Target frame rate for video processing")
    max_latency_sla_ms: float = Field(default=22.0, description="Target SLA max per-frame processing latency in ms")
    
    # Queue and buffer configurations
    capture_queue_maxsize: int = Field(default=2, description="Bounded capture buffer size to prevent frame latency build-up")
    drop_stale_frames: bool = Field(default=True, description="Drop old frames when queue is saturated")
    
    # MediaPipe Pose configurations
    pose_model_complexity: int = Field(default=1, description="0=Fastest/Lightest, 1=Balanced, 2=Highest accuracy")
    smooth_landmarks: bool = Field(default=True, description="Apply 1-Euro jitter filter")
    min_detection_confidence: float = Field(default=0.5, description="Initial detection threshold")
    min_tracking_confidence: float = Field(default=0.5, description="Keypoint tracking threshold")
    
    # Posture sensitivity thresholds
    shoulder_tilt_warning_degrees: float = Field(default=5.0, description="Threshold for uneven shoulder alert")
    spine_deviation_warning_degrees: float = Field(default=10.0, description="Threshold for slumping spine alert")
    hand_fidget_velocity_threshold: float = Field(default=0.04, description="Threshold for excessive hand fidgeting")
    
    # RAG Engine parameters
    rag_chunk_size: int = Field(default=256, description="Token size for transcript chunks")
    rag_chunk_overlap: int = Field(default=32, description="Token overlap between chunks")
    faiss_hnsw_m: int = Field(default=16, description="Number of bi-directional links per node in HNSW graph")
    faiss_ef_construction: int = Field(default=64, description="Exploration depth during index building")
    faiss_ef_search: int = Field(default=32, description="Search exploration depth")
    target_hit_rate_at_3: float = Field(default=0.82, description="Target Hit Rate @ 3 for RAG retrieval")
    
    # Server network settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")

    class Config:
        env_prefix = "PRES_AI_"


settings = PipelineSettings()
