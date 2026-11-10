"""Strongly-typed data definitions for the video pipeline and pose inference worker."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass(slots=True)
class LandmarkPoint:
    """Normalized 3D landmark coordinate with confidence visibility."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass(slots=True)
class KeyLandmarks:
    """Core landmarks extracted for posture and gesture evaluation."""
    left_shoulder: Optional[LandmarkPoint] = None
    right_shoulder: Optional[LandmarkPoint] = None
    mid_shoulder: Optional[LandmarkPoint] = None
    mid_hip: Optional[LandmarkPoint] = None
    left_wrist: Optional[LandmarkPoint] = None
    right_wrist: Optional[LandmarkPoint] = None
    left_elbow: Optional[LandmarkPoint] = None
    right_elbow: Optional[LandmarkPoint] = None
    nose: Optional[LandmarkPoint] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k in self.__slots__:
            pt: Optional[LandmarkPoint] = getattr(self, k)
            if pt is not None:
                result[k] = {"x": round(pt.x, 4), "y": round(pt.y, 4), "z": round(pt.z, 4), "visibility": round(pt.visibility, 3)}
        return result


@dataclass(slots=True)
class PostureMetrics:
    """Instantaneous posture alignment and gesture metrics."""
    shoulder_tilt_degrees: float
    spine_deviation_degrees: float
    hand_movement_velocity: float
    stability_index: float
    is_slumping: bool
    are_shoulders_uneven: bool
    is_fidgeting: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shoulder_tilt_degrees": round(self.shoulder_tilt_degrees, 2),
            "spine_deviation_degrees": round(self.spine_deviation_degrees, 2),
            "hand_movement_velocity": round(self.hand_movement_velocity, 4),
            "stability_index": round(self.stability_index, 1),
            "is_slumping": self.is_slumping,
            "are_shoulders_uneven": self.are_shoulders_uneven,
            "is_fidgeting": self.is_fidgeting
        }


@dataclass(slots=True)
class LatencyBreakdown:
    """Latency profiling across pipeline stages against <22ms SLA."""
    capture_ms: float
    inference_ms: float
    posture_calc_ms: float
    total_latency_ms: float
    sla_satisfied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_ms": round(self.capture_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "posture_calc_ms": round(self.posture_calc_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "sla_satisfied": self.sla_satisfied
        }


@dataclass(slots=True)
class FramePacket:
    """Incoming video frame packet passed through the bounded capture queue."""
    sequence_id: int
    timestamp: float
    frame: np.ndarray


@dataclass(slots=True)
class PoseFrameResult:
    """Full inference result emitted for each processed video frame."""
    sequence_id: int
    timestamp: float
    latency: LatencyBreakdown
    pipeline_fps: float
    landmarks: KeyLandmarks
    posture: PostureMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "timestamp": round(self.timestamp, 4),
            "pipeline_fps": round(self.pipeline_fps, 1),
            "latency": self.latency.to_dict(),
            "landmarks": self.landmarks.to_dict(),
            "posture": self.posture.to_dict()
        }
