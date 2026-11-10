"""Unit tests for the dual-threaded video and pose tracking pipeline."""

import time
import numpy as np
import pytest

from video_pipeline.types import LandmarkPoint, KeyLandmarks, FramePacket, PoseFrameResult
from video_pipeline.posture_calculator import PostureStabilityCalculator
from video_pipeline.camera_stream import FrameCaptureWorker
from video_pipeline.pose_worker import DualThreadedPosePipeline


def test_posture_calculator_upright_baseline():
    """Verify that a centered upright posture yields near-zero deviation and high stability."""
    calc = PostureStabilityCalculator()
    
    # Perfectly level shoulders and vertical spine
    landmarks = KeyLandmarks(
        left_shoulder=LandmarkPoint(x=0.4, y=0.4, z=0.0),
        right_shoulder=LandmarkPoint(x=0.6, y=0.4, z=0.0),
        mid_shoulder=LandmarkPoint(x=0.5, y=0.4, z=0.0),
        mid_hip=LandmarkPoint(x=0.5, y=0.8, z=0.0),
        left_wrist=LandmarkPoint(x=0.3, y=0.6, z=0.0),
        right_wrist=LandmarkPoint(x=0.7, y=0.6, z=0.0)
    )
    
    metrics = calc.calculate(landmarks, timestamp=1.0)
    assert metrics.shoulder_tilt_degrees == pytest.approx(0.0, abs=0.1)
    assert metrics.spine_deviation_degrees == pytest.approx(0.0, abs=0.1)
    assert not metrics.is_slumping
    assert not metrics.are_shoulders_uneven
    assert metrics.stability_index > 90.0


def test_posture_calculator_shoulder_tilt_detection():
    """Verify detection when right shoulder is visibly tilted downwards."""
    calc = PostureStabilityCalculator(shoulder_tilt_threshold_deg=5.0)
    
    # Right shoulder dropped significantly (y=0.45 vs left y=0.35)
    landmarks = KeyLandmarks(
        left_shoulder=LandmarkPoint(x=0.4, y=0.35, z=0.0),
        right_shoulder=LandmarkPoint(x=0.6, y=0.45, z=0.0),
        mid_shoulder=LandmarkPoint(x=0.5, y=0.40, z=0.0),
        mid_hip=LandmarkPoint(x=0.5, y=0.80, z=0.0)
    )
    
    metrics = calc.calculate(landmarks, timestamp=1.0)
    assert metrics.shoulder_tilt_degrees > 20.0
    assert metrics.are_shoulders_uneven is True


def test_frame_capture_worker_drop_stale():
    """Verify that bounded queue drops stale frames to prevent pipeline lag."""
    worker = FrameCaptureWorker(source="EXTERNAL", queue_maxsize=2, drop_stale=True)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Push 5 consecutive frames
    for _ in range(5):
        worker.push_external_frame(dummy_frame)
        
    assert worker.frame_queue.qsize() == 2
    assert worker.total_captured == 5
    assert worker.dropped_frames >= 3


def test_dual_threaded_pose_pipeline_inference():
    """Verify dual-threaded pipeline inference and <22ms latency tracking on frames."""
    pipeline = DualThreadedPosePipeline(
        video_source="EXTERNAL",
        max_latency_sla_ms=22.0,
        drop_stale=True
    )
    
    results = []
    pipeline.register_callback(lambda res: results.append(res))
    
    pipeline.start()
    
    # Feed 10 synthetic frames
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(10):
        pipeline.push_external_frame(dummy_frame)
        time.sleep(0.01)
        
    # Wait for processing
    time.sleep(0.15)
    pipeline.stop()
    
    assert len(results) > 0
    first_res: PoseFrameResult = results[0]
    assert first_res.sequence_id >= 1
    assert first_res.latency.total_latency_ms > 0.0
    assert first_res.posture.stability_index >= 0.0
