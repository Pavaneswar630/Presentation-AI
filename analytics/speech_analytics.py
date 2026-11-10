"""Speech cadence (Words Per Minute) and filler word frequency analytics engine."""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .models import DetectedDefect

logger = logging.getLogger("presentation_ai.analytics.speech")


@dataclass(slots=True)
class SpokenWordToken:
    """Timestamped word token emitted by Whisper or real-time STT pipeline."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0


class SpeechAnalyticsEngine:
    """Analyzes verbal cadence (WPM) and detects filler words and speech disfluencies."""

    COMMON_FILLERS = [
        "um", "uh", "like", "you know", "basically", "actually",
        "sort of", "kind of", "i mean", "right", "so yeah"
    ]

    def __init__(
        self,
        target_wpm_range: Tuple[float, float] = (130.0, 160.0),
        filler_patterns: Optional[List[str]] = None
    ):
        self.target_wpm_min, self.target_wpm_max = target_wpm_range
        patterns = filler_patterns or self.COMMON_FILLERS

        # Compile regex for single and multi-word fillers
        single_words = [p for p in patterns if " " not in p]
        multi_words = [p for p in patterns if " " in p]

        single_re = r"\b(" + "|".join(single_words) + r")\b"
        multi_re = r"\b(" + "|".join([re.escape(m) for m in multi_words]) + r")\b"

        self._single_filler_regex = re.compile(single_re, re.IGNORECASE)
        self._multi_filler_regex = re.compile(multi_re, re.IGNORECASE)

    def analyze_tokens(
        self,
        tokens: List[SpokenWordToken],
        session_duration_sec: float,
        window_size_sec: float = 15.0
    ) -> Dict[str, Any]:
        """Processes timestamped STT tokens to compute WPM time-series and filler metrics."""
        if not tokens or session_duration_sec <= 0:
            return {
                "total_words": 0,
                "average_wpm": 0.0,
                "filler_count": 0,
                "filler_ratio_pct": 0.0,
                "pacing_score": 50.0,
                "wpm_rolling": [],
                "wpm_timestamps": [],
                "defects": []
            }

        total_words = len(tokens)
        duration_minutes = max(session_duration_sec / 60.0, 0.01)
        average_wpm = total_words / duration_minutes

        # 1. Filler Word Counting & Classification
        full_transcript = " ".join([t.word for t in tokens])
        single_matches = self._single_filler_regex.findall(full_transcript)
        multi_matches = self._multi_filler_regex.findall(full_transcript)

        filler_count = len(single_matches) + len(multi_matches)
        filler_ratio_pct = (filler_count / total_words) * 100.0 if total_words > 0 else 0.0

        # 2. Rolling WPM Time-Series (Sliding Window)
        time_steps = np.arange(0.0, session_duration_sec, 5.0)
        wpm_rolling: List[float] = []

        for t_mark in time_steps:
            t_start = max(0.0, t_mark - window_size_sec / 2.0)
            t_end = min(session_duration_sec, t_mark + window_size_sec / 2.0)
            window_dur_min = max((t_end - t_start) / 60.0, 0.001)

            words_in_window = sum(1 for t in tokens if t_start <= t.start_time <= t_end)
            inst_wpm = words_in_window / window_dur_min
            wpm_rolling.append(round(inst_wpm, 1))

        # 3. Pacing Score (0-100) based on target range [130, 160]
        wpm_mid = (self.target_wpm_min + self.target_wpm_max) / 2.0
        wpm_delta = abs(average_wpm - wpm_mid)
        # 100 at 145 WPM, falls off gracefully if too fast or slow
        pacing_score = max(0.0, min(100.0, 100.0 - (wpm_delta * 1.5)))

        # 4. Speech Defect Detection
        defects: List[DetectedDefect] = []

        # High filler cluster check
        window_fillers = 0
        window_start_idx = 0
        for i, tok in enumerate(tokens):
            while tok.start_time - tokens[window_start_idx].start_time > 15.0:
                if self._is_filler(tokens[window_start_idx].word):
                    window_fillers = max(0, window_fillers - 1)
                window_start_idx += 1

            if self._is_filler(tok.word):
                window_fillers += 1

            if window_fillers >= 4:
                defects.append(DetectedDefect(
                    defect_type="FILLER_WORD_CLUSTER",
                    timestamp_range=(round(tokens[window_start_idx].start_time, 1), round(tok.end_time, 1)),
                    severity="MODERATE",
                    metric_value=f"Cluster of {window_fillers} fillers detected within 15 seconds"
                ))
                window_fillers = 0  # Reset cluster counter to avoid duplicate triggers

        # Excessive speed check
        if average_wpm > 175.0:
            defects.append(DetectedDefect(
                defect_type="RAPID_PACING",
                timestamp_range=(0.0, round(session_duration_sec, 1)),
                severity="HIGH",
                metric_value=f"Average speaking cadence of {average_wpm:.1f} WPM exceeded 175 WPM maximum"
            ))
        elif average_wpm < 110.0 and total_words > 20:
            defects.append(DetectedDefect(
                defect_type="SLOW_PACING",
                timestamp_range=(0.0, round(session_duration_sec, 1)),
                severity="LOW",
                metric_value=f"Average speaking cadence of {average_wpm:.1f} WPM fell below 110 WPM minimum"
            ))

        return {
            "total_words": total_words,
            "average_wpm": round(average_wpm, 1),
            "filler_count": filler_count,
            "filler_ratio_pct": round(filler_ratio_pct, 2),
            "pacing_score": round(pacing_score, 1),
            "wpm_rolling": wpm_rolling,
            "wpm_timestamps": [round(float(t), 1) for t in time_steps],
            "defects": defects
        }

    def _is_filler(self, word: str) -> bool:
        """Checks if a single word is in the filler set."""
        return bool(self._single_filler_regex.match(word.strip()))
