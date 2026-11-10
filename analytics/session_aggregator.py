"""Session data aggregator synthesizing posture, speech, and RAG coaching insights into PostSessionReport."""

import logging
from typing import List, Optional, Dict, Any
from .models import (
    PostSessionReport,
    SessionSummary,
    RadarDimensions,
    TimeSeriesMetrics,
    DetectedDefect,
    CoachingRecommendation,
    TelemetryPoint
)
from .posture_analytics import PostureAnalyticsEngine
from .speech_analytics import SpeechAnalyticsEngine, SpokenWordToken
from rag_engine.retriever import CoachingRetriever

logger = logging.getLogger("presentation_ai.analytics.aggregator")


class SessionDataAggregator:
    """Ingests multi-modal session telemetry and compiles a final structured evaluation report."""

    def __init__(
        self,
        retriever: Optional[CoachingRetriever] = None,
        posture_engine: Optional[PostureAnalyticsEngine] = None,
        speech_engine: Optional[SpeechAnalyticsEngine] = None
    ):
        self.retriever = retriever
        self.posture_engine = posture_engine or PostureAnalyticsEngine()
        self.speech_engine = speech_engine or SpeechAnalyticsEngine()

    def generate_report(
        self,
        session_id: str,
        duration_seconds: float,
        telemetry: List[TelemetryPoint],
        spoken_tokens: List[SpokenWordToken]
    ) -> PostSessionReport:
        """Processes raw telemetry and produces a validated PostSessionReport."""
        # 1. Posture Analytics
        posture_score, rolling_stability, rolling_hand_energy = self.posture_engine.compute_stability_scores(
            telemetry=telemetry
        )
        posture_defects = self.posture_engine.detect_defects(telemetry=telemetry)

        # 2. Speech Analytics
        speech_results = self.speech_engine.analyze_tokens(
            tokens=spoken_tokens,
            session_duration_sec=duration_seconds
        )
        speech_defects = speech_results["defects"]

        # Combined defects
        all_defects: List[DetectedDefect] = posture_defects + speech_defects

        # 3. Time Series Interpolation
        timestamps = [round(p.timestamp, 1) for p in telemetry]
        wpm_rolling = speech_results["wpm_rolling"]

        # Ensure consistent lengths or downsample for charting
        time_series = TimeSeriesMetrics(
            timestamps=timestamps[:len(rolling_stability)],
            wpm_rolling=wpm_rolling,
            posture_stability_rolling=rolling_stability,
            hand_fidget_energy=rolling_hand_energy
        )

        # 4. Multi-Dimensional Radar Scoring
        pacing_score = speech_results["pacing_score"]
        filler_penalty = min(50.0, speech_results["filler_ratio_pct"] * 10.0)
        clarity_score = max(0.0, min(100.0, 100.0 - filler_penalty))

        # Gesture expressiveness: penalize high jitter and zero movement
        mean_hand = float(sum(rolling_hand_energy) / len(rolling_hand_energy)) if rolling_hand_energy else 0.015
        gesture_score = max(0.0, min(100.0, 100.0 - abs(mean_hand - 0.018) * 2000.0))

        # Energy consistency: derived from rolling variance
        energy_score = max(0.0, min(100.0, (posture_score * 0.5) + (pacing_score * 0.5)))

        radar = RadarDimensions(
            posture=round(posture_score, 1),
            pacing=round(pacing_score, 1),
            gesture_expressiveness=round(gesture_score, 1),
            verbal_clarity=round(clarity_score, 1),
            energy_consistency=round(energy_score, 1)
        )

        # Overall composite score
        overall_score = round(
            (radar.posture * 0.25) +
            (radar.pacing * 0.25) +
            (radar.verbal_clarity * 0.25) +
            (radar.gesture_expressiveness * 0.15) +
            (radar.energy_consistency * 0.10),
            1
        )

        summary = SessionSummary(
            overall_score=overall_score,
            duration_seconds=round(duration_seconds, 2),
            total_words_spoken=speech_results["total_words"],
            average_cadence_wpm=speech_results["average_wpm"],
            filler_word_count=speech_results["filler_count"],
            filler_word_ratio_pct=speech_results["filler_ratio_pct"],
            posture_stability_score=round(posture_score, 1)
        )

        # 5. RAG Coaching Enrichment
        recommendations: List[CoachingRecommendation] = []
        if self.retriever is not None and all_defects:
            seen_topics = set()
            for defect in all_defects:
                advice_list = self.retriever.retrieve_for_defect(
                    defect_type=defect.defect_type,
                    context_details=defect.metric_value,
                    top_k=2
                )
                for advice in advice_list:
                    if advice.topic not in seen_topics:
                        seen_topics.add(advice.topic)
                        recommendations.append(CoachingRecommendation(
                            source_transcript_id=advice.source_chunk_id,
                            similarity_score=round(advice.similarity_score, 3),
                            topic=advice.topic,
                            recommendation=advice.recommendation
                        ))
                    if len(recommendations) >= 3:
                        break
                if len(recommendations) >= 3:
                    break

        return PostSessionReport(
            session_id=session_id,
            status="COMPLETED",
            summary=summary,
            radar_dimensions=radar,
            time_series_metrics=time_series,
            detected_defects=all_defects,
            rag_coaching_recommendations=recommendations
        )
