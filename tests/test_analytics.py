"""Unit tests for the Post-Session Analytics Engine."""

import pytest
from analytics.models import TelemetryPoint, PostSessionReport
from analytics.posture_analytics import PostureAnalyticsEngine
from analytics.speech_analytics import SpeechAnalyticsEngine, SpokenWordToken
from analytics.session_aggregator import SessionDataAggregator
from rag_engine.chunker import TranscriptChunker, TranscriptDocument
from rag_engine.indexer import FAISSHNSWIndexer
from rag_engine.retriever import CoachingRetriever


def test_posture_stability_tensor_scoring():
    """Verify posture stability score decreases with high angular variance."""
    engine = PostureAnalyticsEngine()

    # Steady, level posture
    steady_telemetry = [
        TelemetryPoint(timestamp=float(i), shoulder_tilt_degrees=1.0, spine_deviation_degrees=2.0, hand_movement_velocity=0.01)
        for i in range(30)
    ]
    steady_score, rolling_s, _ = engine.compute_stability_scores(steady_telemetry)
    assert steady_score > 90.0

    # Highly erratic, slumped posture
    erratic_telemetry = [
        TelemetryPoint(
            timestamp=float(i),
            shoulder_tilt_degrees=float(10.0 + (i % 8)),
            spine_deviation_degrees=float(15.0 + (i % 10)),
            hand_movement_velocity=0.06
        )
        for i in range(30)
    ]
    erratic_score, _, _ = engine.compute_stability_scores(erratic_telemetry)
    assert erratic_score < steady_score
    assert erratic_score < 75.0


def test_posture_defect_detection():
    """Verify sustained slumping triggers SLUMPED_POSTURE defect."""
    engine = PostureAnalyticsEngine(slump_threshold_deg=12.0)

    # 10 seconds of severe slumping
    slump_telemetry = [
        TelemetryPoint(timestamp=float(i), spine_deviation_degrees=16.5)
        for i in range(10)
    ]
    defects = engine.detect_defects(slump_telemetry, min_duration_sec=3.0)
    assert len(defects) >= 1
    assert defects[0].defect_type == "SLUMPED_POSTURE"
    assert defects[0].severity in ["MODERATE", "HIGH"]


def test_speech_cadence_and_filler_frequency():
    """Verify WPM and filler word calculation."""
    engine = SpeechAnalyticsEngine()

    # 30 words in 12 seconds = 150 WPM (perfect cadence)
    words = "Good morning everyone today we will discuss our strategic vision for the upcoming fiscal quarter with um exciting milestones and uh great achievements for like our whole team".split()
    tokens = [
        SpokenWordToken(word=w, start_time=i * 0.4, end_time=(i * 0.4) + 0.35)
        for i, w in enumerate(words)
    ]

    results = engine.analyze_tokens(tokens=tokens, session_duration_sec=12.0)
    assert results["total_words"] == len(words)
    assert 130.0 <= results["average_wpm"] <= 165.0
    # Expected fillers: "um", "uh", "like" (3 fillers)
    assert results["filler_count"] >= 3
    assert results["filler_ratio_pct"] > 5.0
    assert results["pacing_score"] > 85.0


def test_session_aggregator_report_generation():
    """Verify end-to-end report synthesis with RAG coaching tips."""
    # Build a mini RAG index for coaching advice
    indexer = FAISSHNSWIndexer()
    chunker = TranscriptChunker()
    doc = TranscriptDocument(
        transcript_id="doc_posture",
        title="Spine Alignment",
        speaker="Coach",
        topic="Upper Torso Alignment & Spine Posture",
        content="Keep your spine straight and chest open to project executive presence."
    )
    indexer.add_chunks(chunker.chunk_corpus([doc]))
    retriever = CoachingRetriever(indexer)

    aggregator = SessionDataAggregator(retriever=retriever)

    # Telemetry with sustained slumping
    telemetry = [
        TelemetryPoint(timestamp=float(i), spine_deviation_degrees=15.0, shoulder_tilt_degrees=1.0)
        for i in range(10)
    ]
    tokens = [
        SpokenWordToken(word="hello", start_time=0.0, end_time=0.5),
        SpokenWordToken(word="world", start_time=1.0, end_time=1.5)
    ]

    report: PostSessionReport = aggregator.generate_report(
        session_id="test_session_123",
        duration_seconds=10.0,
        telemetry=telemetry,
        spoken_tokens=tokens
    )

    assert report.session_id == "test_session_123"
    assert report.status == "COMPLETED"
    assert 0.0 <= report.summary.overall_score <= 100.0
    assert report.radar_dimensions.posture >= 0.0
    assert len(report.detected_defects) > 0
    assert len(report.rag_coaching_recommendations) > 0
