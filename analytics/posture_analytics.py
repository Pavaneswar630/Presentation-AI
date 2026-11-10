"""PyTorch and NumPy vectorized posture stability scoring and defect analysis."""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .models import TelemetryPoint, DetectedDefect

logger = logging.getLogger("presentation_ai.analytics.posture")


class PostureAnalyticsEngine:
    """Computes session-wide posture stability, kinetic energy, and detects body language defects."""

    def __init__(
        self,
        alpha_shoulder_var: float = 0.03,
        beta_spine_var: float = 0.05,
        gamma_hand_energy: float = 3.0,
        slump_threshold_deg: float = 12.0,
        shoulder_tilt_threshold_deg: float = 6.0,
        fidget_threshold: float = 0.035
    ):
        self.alpha = alpha_shoulder_var
        self.beta = beta_spine_var
        self.gamma = gamma_hand_energy
        self.slump_threshold_deg = slump_threshold_deg
        self.shoulder_tilt_threshold_deg = shoulder_tilt_threshold_deg
        self.fidget_threshold = fidget_threshold

    def compute_stability_scores(
        self,
        telemetry: List[TelemetryPoint],
        window_size: int = 15
    ) -> Tuple[float, List[float], List[float]]:
        """Calculates global posture stability score and rolling time-series using PyTorch/NumPy.
        
        Returns:
            global_score: 0-100 posture stability score.
            rolling_stability: Array of rolling stability scores.
            hand_energy: Array of smoothed kinetic hand energy values.
        """
        if not telemetry:
            return 100.0, [], []

        # Extract numpy arrays
        tilt_arr = np.array([p.shoulder_tilt_degrees for p in telemetry], dtype=np.float32)
        spine_arr = np.array([p.spine_deviation_degrees for p in telemetry], dtype=np.float32)
        hand_arr = np.array([p.hand_movement_velocity for p in telemetry], dtype=np.float32)
        n = len(telemetry)

        # PyTorch acceleration when available
        if torch is not None:
            t_tilt = torch.from_numpy(tilt_arr)
            t_spine = torch.from_numpy(spine_arr)
            t_hand = torch.from_numpy(hand_arr)

            var_tilt = float(torch.var(t_tilt).item()) if n > 1 else 0.0
            var_spine = float(torch.var(t_spine).item()) if n > 1 else 0.0
            mean_hand = float(torch.mean(t_hand).item())
        else:
            var_tilt = float(np.var(tilt_arr)) if n > 1 else 0.0
            var_spine = float(np.var(spine_arr)) if n > 1 else 0.0
            mean_hand = float(np.mean(hand_arr))

        # Exponential penalty equation: S = 100 * exp(-(alpha*var_tilt + beta*var_spine + gamma*mean_hand))
        exponent = (self.alpha * var_tilt) + (self.beta * var_spine) + (self.gamma * mean_hand)
        global_score = float(np.clip(100.0 * np.exp(-exponent), 0.0, 100.0))

        # Compute rolling window time-series for UI visualization
        rolling_stability: List[float] = []
        rolling_hand_energy: List[float] = []

        half_w = max(1, window_size // 2)
        for i in range(n):
            start_i = max(0, i - half_w)
            end_i = min(n, i + half_w + 1)

            w_tilt = tilt_arr[start_i:end_i]
            w_spine = spine_arr[start_i:end_i]
            w_hand = hand_arr[start_i:end_i]

            w_var_t = float(np.var(w_tilt)) if len(w_tilt) > 1 else 0.0
            w_var_s = float(np.var(w_spine)) if len(w_spine) > 1 else 0.0
            w_mean_h = float(np.mean(w_hand))

            w_exp = (self.alpha * w_var_t) + (self.beta * w_var_s) + (self.gamma * w_mean_h)
            w_score = float(np.clip(100.0 * np.exp(-w_exp), 0.0, 100.0))

            rolling_stability.append(round(w_score, 1))
            rolling_hand_energy.append(round(w_mean_h, 4))

        return round(global_score, 1), rolling_stability, rolling_hand_energy

    def detect_defects(self, telemetry: List[TelemetryPoint], min_duration_sec: float = 3.0) -> List[DetectedDefect]:
        """Scans session telemetry to identify sustained posture and movement defects."""
        if not telemetry:
            return []

        defects: List[DetectedDefect] = []

        def track_interval(condition_fn, defect_type: str, severity_rule_fn):
            in_defect = False
            start_t = 0.0
            peak_val = 0.0

            for p in telemetry:
                cond, val = condition_fn(p)
                if cond:
                    if not in_defect:
                        in_defect = True
                        start_t = p.timestamp
                        peak_val = val
                    else:
                        peak_val = max(peak_val, val)
                else:
                    if in_defect:
                        duration = p.timestamp - start_t
                        if duration >= min_duration_sec:
                            severity, msg = severity_rule_fn(duration, peak_val)
                            defects.append(DetectedDefect(
                                defect_type=defect_type,
                                timestamp_range=(round(start_t, 1), round(p.timestamp, 1)),
                                severity=severity,
                                metric_value=msg
                            ))
                        in_defect = False

            # Check if active at session close
            if in_defect:
                duration = telemetry[-1].timestamp - start_t
                if duration >= min_duration_sec:
                    severity, msg = severity_rule_fn(duration, peak_val)
                    defects.append(DetectedDefect(
                        defect_type=defect_type,
                        timestamp_range=(round(start_t, 1), round(telemetry[-1].timestamp, 1)),
                        severity=severity,
                        metric_value=msg
                    ))

        # 1. Slumped spine
        track_interval(
            condition_fn=lambda p: (p.spine_deviation_degrees > self.slump_threshold_deg, p.spine_deviation_degrees),
            defect_type="SLUMPED_POSTURE",
            severity_rule_fn=lambda dur, val: (
                "HIGH" if val > 18.0 or dur > 10.0 else "MODERATE",
                f"Spine tilt deviation peaked at {val:.1f}° for {dur:.1f}s"
            )
        )

        # 2. Uneven shoulder tilt
        track_interval(
            condition_fn=lambda p: (p.shoulder_tilt_degrees > self.shoulder_tilt_threshold_deg, p.shoulder_tilt_degrees),
            defect_type="SHOULDER_ASYMMETRY",
            severity_rule_fn=lambda dur, val: (
                "HIGH" if val > 10.0 else "LOW",
                f"Shoulder tilt asymmetry reached {val:.1f}° for {dur:.1f}s"
            )
        )

        # 3. Excessive hand fidgeting
        track_interval(
            condition_fn=lambda p: (p.hand_movement_velocity > self.fidget_threshold, p.hand_movement_velocity),
            defect_type="EXCESSIVE_FIDGETING",
            severity_rule_fn=lambda dur, val: (
                "MODERATE",
                f"Hand kinetic jitter velocity peaked at {val:.4f} units/s for {dur:.1f}s"
            )
        )

        return defects
