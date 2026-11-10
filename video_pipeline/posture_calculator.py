"""Posture trigonometry and geometric stability calculator."""

import math
from typing import Optional, Tuple
from .types import LandmarkPoint, KeyLandmarks, PostureMetrics


class PostureStabilityCalculator:
    """Computes real-time body language geometry, spine alignment, and kinetic jitter."""

    def __init__(
        self,
        shoulder_tilt_threshold_deg: float = 5.0,
        spine_deviation_threshold_deg: float = 10.0,
        hand_velocity_threshold: float = 0.04
    ):
        self.shoulder_tilt_threshold_deg = shoulder_tilt_threshold_deg
        self.spine_deviation_threshold_deg = spine_deviation_threshold_deg
        self.hand_velocity_threshold = hand_velocity_threshold

        # Historical state for kinetic velocity tracking
        self._prev_left_wrist: Optional[Tuple[float, float]] = None
        self._prev_right_wrist: Optional[Tuple[float, float]] = None
        self._prev_timestamp: Optional[float] = None

    def calculate(
        self,
        landmarks: KeyLandmarks,
        timestamp: float
    ) -> PostureMetrics:
        """Evaluates geometric posture parameters and movement energy."""
        
        # 1. Shoulder Tilt Angle
        shoulder_tilt_deg = 0.0
        if landmarks.left_shoulder and landmarks.right_shoulder:
            dx = landmarks.right_shoulder.x - landmarks.left_shoulder.x
            dy = landmarks.right_shoulder.y - landmarks.left_shoulder.y
            # In image space, positive angle indicates right shoulder is lower than left
            shoulder_tilt_deg = math.degrees(math.atan2(dy, dx))
            # Standardize relative to horizontal line (0 degrees)
            shoulder_tilt_deg = abs(shoulder_tilt_deg)

        # 2. Spine Alignment / Plumb-line Deviation
        spine_dev_deg = 0.0
        if landmarks.mid_shoulder and landmarks.mid_hip:
            # Spine vector pointing upward from hip to shoulder:
            # mid_shoulder.y is smaller than mid_hip.y in image coords
            vx = landmarks.mid_shoulder.x - landmarks.mid_hip.x
            vy = landmarks.mid_shoulder.y - landmarks.mid_hip.y
            mag = math.hypot(vx, vy)
            if mag > 1e-5:
                # Vertical upward vector in screen coords is (0, -1)
                # Dot product: vx*0 + vy*(-1) = -vy
                cos_angle = max(-1.0, min(1.0, (-vy) / mag))
                spine_dev_deg = math.degrees(math.acos(cos_angle))
        elif landmarks.mid_shoulder and landmarks.nose:
            # Fallback when hips are occluded: evaluate neck-to-nose vertical vector
            vx = landmarks.nose.x - landmarks.mid_shoulder.x
            vy = landmarks.nose.y - landmarks.mid_shoulder.y
            mag = math.hypot(vx, vy)
            if mag > 1e-5:
                cos_angle = max(-1.0, min(1.0, (-vy) / mag))
                spine_dev_deg = math.degrees(math.acos(cos_angle))

        # 3. Hand Gesture & Fidgeting Velocity
        hand_velocity = 0.0
        dt = (timestamp - self._prev_timestamp) if self._prev_timestamp is not None else 0.033
        dt = max(dt, 0.001)  # Guard against division by zero

        v_l = 0.0
        if landmarks.left_wrist and self._prev_left_wrist is not None:
            dist_l = math.hypot(
                landmarks.left_wrist.x - self._prev_left_wrist[0],
                landmarks.left_wrist.y - self._prev_left_wrist[1]
            )
            v_l = dist_l / dt

        v_r = 0.0
        if landmarks.right_wrist and self._prev_right_wrist is not None:
            dist_r = math.hypot(
                landmarks.right_wrist.x - self._prev_right_wrist[0],
                landmarks.right_wrist.y - self._prev_right_wrist[1]
            )
            v_r = dist_r / dt

        hand_velocity = (v_l + v_r) / 2.0

        # Update historical cache
        if landmarks.left_wrist:
            self._prev_left_wrist = (landmarks.left_wrist.x, landmarks.left_wrist.y)
        if landmarks.right_wrist:
            self._prev_right_wrist = (landmarks.right_wrist.x, landmarks.right_wrist.y)
        self._prev_timestamp = timestamp

        # 4. Instantaneous Stability Index (0 - 100)
        # Penalties: shoulder tilt, spine deviation, excessive jitter
        penalty = (shoulder_tilt_deg * 2.0) + (spine_dev_deg * 2.5) + (hand_velocity * 120.0)
        stability_index = max(0.0, min(100.0, 100.0 - penalty))

        # 5. Threshold triggers
        are_shoulders_uneven = shoulder_tilt_deg > self.shoulder_tilt_threshold_deg
        is_slumping = spine_dev_deg > self.spine_deviation_threshold_deg
        is_fidgeting = hand_velocity > self.hand_velocity_threshold

        return PostureMetrics(
            shoulder_tilt_degrees=shoulder_tilt_deg,
            spine_deviation_degrees=spine_dev_deg,
            hand_movement_velocity=hand_velocity,
            stability_index=stability_index,
            is_slumping=is_slumping,
            are_shoulders_uneven=are_shoulders_uneven,
            is_fidgeting=is_fidgeting
        )
