"""Unit and integration tests for dynamic question generation and interactive Q&A."""

import json
import pytest
from starlette.testclient import TestClient

from rag_engine.question_generator import (
    TranscriptSegmentAnalyzer,
    LLMQuestionGenerator,
    PresentationQuestion,
    QuestionGenerationResult
)
from rag_engine.chunker import TranscriptChunker, TranscriptDocument
from rag_engine.indexer import FAISSHNSWIndexer
from rag_engine.retriever import CoachingRetriever
from api.app import create_app
from api.routes import set_global_session_manager
from api.session_manager import SessionManager
from analytics.speech_analytics import SpokenWordToken


def test_transcript_segment_analyzer_metrics_and_claims():
    """Verify that analyzer extracts quantifiable metrics and core assertions from speech."""
    analyzer = TranscriptSegmentAnalyzer()
    
    transcript = (
        "Good morning everyone. Today we built a new distributed architecture. "
        "We believe our solution reduced latency by 35% and increased throughput by 4x. "
        "Our strategy fundamentally outperforms legacy platforms, generating over $2.5 million in cost savings."
    )
    
    analysis = analyzer.analyze(transcript)
    assert analysis["word_count"] > 25
    assert len(analysis["metrics_claimed"]) >= 2
    # Check for extracted metrics
    metrics_str = " ".join(analysis["metrics_claimed"])
    assert "35%" in metrics_str or "4x" in metrics_str or "$2.5" in metrics_str
    
    # Check for extracted claims
    assert len(analysis["core_claims"]) >= 1
    assert "outperforms" in " ".join(analysis["core_claims"]) or "reduced" in " ".join(analysis["core_claims"])
    
    # Check detected topics
    assert "architecture" in analysis["topics_detected"] or "latency" in analysis["topics_detected"] or "throughput" in analysis["topics_detected"]


def test_llm_question_generator_with_rag_context():
    """Verify generation of 3 targeted evaluator questions and live probes using RAG context."""
    import asyncio
    # Build test RAG index
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    chunker = TranscriptChunker()
    doc = TranscriptDocument(
        transcript_id="doc_scale",
        title="Scalable Systems",
        speaker="Coach",
        topic="Scalability & Metrics",
        content="When presenting technical metrics like latency reduction and throughput, always anchor against baseline numbers."
    )
    indexer.add_chunks(chunker.chunk_corpus([doc]))
    
    gen = LLMQuestionGenerator(indexer=indexer)
    
    transcript = (
        "We deployed an event-driven microservices pipeline that reduced memory footprint by 40%. "
        "We believe this framework provides a competitive edge in mission-critical banking transactions."
    )
    
    result: QuestionGenerationResult = asyncio.run(gen.generate_questions(
        session_id="sess_q_test",
        transcript_text=transcript,
        question_count=3,
        difficulty="probing"
    ))
    
    assert result.session_id == "sess_q_test"
    assert len(result.targeted_questions) == 3
    
    q1 = result.targeted_questions[0]
    assert isinstance(q1, PresentationQuestion)
    assert len(q1.question_text) > 15
    assert q1.category in ["Critical Thinking", "Clarification", "Feasibility", "Business Impact"]
    assert len(q1.rationale) > 10
    assert len(q1.suggested_answer_focus) > 10
    
    # Verify live probe prompt
    assert len(result.live_probes) >= 1
    assert "probe" in result.live_probes[0].prompt_text.lower() or "?" in result.live_probes[0].prompt_text


def test_generator_accepts_spoken_tokens_and_requested_count(monkeypatch):
    """Verify tokenized transcripts work offline and the requested count is honored."""
    import asyncio

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = LLMQuestionGenerator()
    tokens = [
        SpokenWordToken(word, index * 0.3, (index + 1) * 0.3)
        for index, word in enumerate(
            "We reduced latency by 35 percent and improved customer retention".split()
        )
    ]

    result = asyncio.run(generator.generate_questions(
        session_id="sess_tokens",
        transcript_text=tokens,
        question_count=5
    ))

    assert generator.analyzer.analyze(tokens)["word_count"] == len(tokens)
    assert len(result.targeted_questions) == 5
    assert result.targeted_questions[3].category == "Clarification"
    assert result.live_probes


def test_openai_compatible_provider_uses_async_client(monkeypatch):
    """Verify an explicitly configured OpenAI-compatible provider is called asynchronously."""
    import asyncio

    questions = [
        {
            "question_text": f"Question {index}?",
            "category": "Critical Thinking",
            "rationale": "Tests the underlying claim.",
            "suggested_answer_focus": "Support the answer with evidence.",
            "difficulty": "probing"
        }
        for index in range(3)
    ]
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"questions": questions})}}]}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            calls.append((url, headers, json))
            return FakeResponse()

    monkeypatch.setattr("rag_engine.question_generator.httpx.AsyncClient", FakeAsyncClient)
    generator = LLMQuestionGenerator(
        api_key="test-key",
        provider="openai",
        base_url="https://llm.example/v1"
    )
    result = asyncio.run(generator.generate_questions(
        session_id="sess_openai",
        transcript_text="We improved customer retention by 20%.",
        question_count=3
    ))

    assert len(result.targeted_questions) == 3
    assert calls[0][0] == "https://llm.example/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer test-key"


@pytest.fixture(scope="module")
def api_client():
    """Prepares a test client with session manager and RAG index."""
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    chunker = TranscriptChunker()
    doc = TranscriptDocument(
        transcript_id="doc_keynote",
        title="Enterprise Keynote",
        speaker="Coach",
        topic="Executive Strategy",
        content="Executive audiences expect clear ROI metrics and risk mitigation contingencies."
    )
    indexer.add_chunks(chunker.chunk_corpus([doc]))
    retriever = CoachingRetriever(indexer)
    
    manager = SessionManager(retriever=retriever)
    set_global_session_manager(manager)
    
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_rest_generate_questions_endpoint(api_client: TestClient):
    """Verify POST /api/v1/sessions/{session_id}/generate-questions returns 3 targeted Q&A questions."""
    # 1. Start a session
    start_res = api_client.post("/api/v1/sessions/start", json={"presentation_title": "AI Cloud Platform"})
    assert start_res.status_code == 201
    session_id = start_res.json()["session_id"]
    
    # 2. Call generate-questions endpoint
    q_res = api_client.post(
        f"/api/v1/sessions/{session_id}/generate-questions",
        json={"question_count": 3, "difficulty": "challenging"}
    )
    assert q_res.status_code == 200
    data = q_res.json()
    assert data["session_id"] == session_id
    assert len(data["targeted_questions"]) == 3
    assert len(data["live_probes"]) >= 1
    
    first_q = data["targeted_questions"][0]
    assert "question_text" in first_q
    assert "category" in first_q
    assert "suggested_answer_focus" in first_q

    default_res = api_client.post(f"/api/v1/sessions/{session_id}/generate-questions")
    assert default_res.status_code == 200
    assert len(default_res.json()["targeted_questions"]) == 3

    invalid_res = api_client.post(
        f"/api/v1/sessions/{session_id}/generate-questions",
        json={"difficulty": "expert"}
    )
    assert invalid_res.status_code == 422


def test_websocket_request_live_probes(api_client: TestClient):
    """Verify WebSocket responds to request_live_probes with live_probe_question event."""
    session_id = "sess_probe_ws_test"
    
    with api_client.websocket_connect(f"/ws/v1/session/{session_id}") as ws:
        # Consume connection_ack
        _ = ws.receive_text()

        ws.send_text(json.dumps({"type": "audio_packet", "word": "35%"}))
        # Request live probe
        ws.send_text(json.dumps({"type": "request_live_probes"}))
        msg_raw = ws.receive_text()
        msg = json.loads(msg_raw)
        
        assert msg["event"] == "live_probe_question"
        assert "probe" in msg
        assert "prompt_text" in msg["probe"]
        assert len(msg["probe"]["prompt_text"]) > 10
        assert msg["probe"]["trigger_phrase"] == "35%"
