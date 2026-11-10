"""Integration tests for FastAPI REST routes and WebSocket streaming."""

import base64
import json
import numpy as np
import pytest
from starlette.testclient import TestClient

from api.app import create_app
from api.routes import set_global_session_manager
from api.session_manager import SessionManager
from rag_engine.corpus_generator import TranscriptCorpusGenerator
from rag_engine.chunker import TranscriptChunker
from rag_engine.indexer import FAISSHNSWIndexer
from rag_engine.retriever import CoachingRetriever


@pytest.fixture(scope="module")
def client():
    """Builds a test client with pre-warmed RAG index and session manager."""
    corpus, _ = TranscriptCorpusGenerator.generate_corpus(count=100)
    chunker = TranscriptChunker()
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    indexer.add_chunks(chunker.chunk_corpus(corpus))
    retriever = CoachingRetriever(indexer)

    manager = SessionManager(retriever=retriever)
    set_global_session_manager(manager)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /api/v1/health returns 200 OK and subsystem readiness."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "video_pipeline" in data["components"]
    assert "rag_engine" in data["components"]


def test_ui_index_served(client: TestClient):
    """Verify GET / serves the interactive coaching UI index.html."""
    res = client.get("/")
    assert res.status_code == 200
    assert "Presentation AI" in res.text
    assert "radarChart" in res.text


def test_session_lifecycle_rest_api(client: TestClient):
    """Verify complete session lifecycle: start -> status -> stop -> report."""
    # 1. Start Session
    start_payload = {
        "user_id": "usr_test_42",
        "presentation_title": "Quarterly Product Review",
        "config": {
            "target_fps": 30.0,
            "max_latency_sla_ms": 22.0
        }
    }
    start_res = client.post("/api/v1/sessions/start", json=start_payload)
    assert start_res.status_code == 201
    start_data = start_res.json()
    session_id = start_data["session_id"]
    assert session_id.startswith("sess_")
    assert start_data["status"] == "ACTIVE"

    # 2. Check Session Status
    status_res = client.get(f"/api/v1/sessions/{session_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["session_id"] == session_id
    assert status_data["status"] == "ACTIVE"

    # 3. Stop Session
    stop_res = client.post(f"/api/v1/sessions/{session_id}/stop")
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["status"] == "COMPLETED"

    # 4. Fetch PostSessionReport
    report_res = client.get(f"/api/v1/sessions/{session_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert report_data["session_id"] == session_id
    assert report_data["status"] == "COMPLETED"
    assert "summary" in report_data
    assert "radar_dimensions" in report_data


def test_websocket_handshake_and_streaming(client: TestClient):
    """Verify WebSocket connection, connection_ack, heartbeat, and frame processing."""
    session_id = "sess_ws_test"

    with client.websocket_connect(f"/ws/v1/session/{session_id}") as ws:
        # 1. Receive connection_ack
        ack_raw = ws.receive_text()
        ack = json.loads(ack_raw)
        assert ack["event"] == "connection_ack"
        assert ack["session_id"] == session_id

        # 2. Test Client Heartbeat -> Server Pong
        ws.send_text(json.dumps({"type": "client_heartbeat"}))
        pong_raw = ws.receive_text()
        pong = json.loads(pong_raw)
        assert pong["event"] == "server_pong"

        # 3. Send frame packet (encoded small black image)
        raw_pixels = b"\x00" * (100 * 100 * 3)
        b64_img = base64.b64encode(raw_pixels).decode("utf-8")
        frame_msg = {
            "type": "frame_packet",
            "seq_id": 1,
            "data": b64_img
        }
        ws.send_text(json.dumps(frame_msg))
