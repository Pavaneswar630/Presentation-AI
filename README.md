# Presentation AI: Real-Time Speech & Body Language Coaching System

Presentation AI is an asynchronous, low-latency AI coaching engine designed to analyze physical body language (posture, spinal alignment, gesture jitter) and vocal delivery (cadence, words-per-minute, filler word frequency) in real-time.

---

## Key Features & Architecture Highlights

* **Dual-Threaded Video Pipeline (`video_pipeline/`):**
  * Decouples frame capture from MediaPipe Pose neural network inference via dedicated OS threads.
  * Bounded ring-buffer queue (`maxsize=2`) with stale-frame discard to eliminate buffer latency accumulation.
  * Sub-22ms per-frame processing latency SLA at a steady 30 FPS.
  * Trigonometric evaluation of shoulder tilt angle, spine plumb-line deviation, and hand kinetic velocity.

* **Offline RAG Engine (`rag_engine/`):**
  * Ingests 1,000 reference presentation transcripts.
  * 256-token sliding window chunking with 32-token overlap.
  * FAISS HNSW index ($M=16, efConstruction=64, efSearch=32$) with `BAAI/bge-large-en-v1.5` embeddings (1024-d).
  * Benchmark evaluation harness targeting $\ge 0.82$ Hit Rate@3.

* **Post-Session PyTorch Analytics Engine (`analytics/`):**
  * PyTorch/NumPy vectorized temporal stability evaluation.
  * Rolling window Words Per Minute (WPM) cadence estimation and filler word ratio calculation.
  * Outputs a structured JSON report with radar dimension scoring and actionable coaching tips.

* **FastAPI & WebSocket Streaming Gateway (`api/`):**
  * Asynchronous REST lifecycle endpoints (`/start`, `/stop`, `/report`, `/health`).
  * Real-time full-duplex WebSocket stream (`/ws/v1/session/{session_id}`) with live HUD telemetry alerts.

---

## Directory Structure

```text
presentation_ai/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py              # Pydantic Settings (SLA thresholds, model paths, ports)
├── video_pipeline/
│   ├── __init__.py
│   ├── types.py                 # Data models: LandmarkPoint, PoseFrameResult, Metrics
│   ├── camera_stream.py         # Capture worker thread & drop-frame ring buffer
│   ├── posture_calculator.py    # Math module for spine/shoulder/hand stability
│   └── pose_worker.py           # Dual-threaded pose estimation & landmark extractor
├── rag_engine/
│   ├── __init__.py
│   ├── chunker.py               # 256-token chunker with 32-token overlap
│   ├── indexer.py               # FAISS HNSW (M=16) + BGE-large-en vector store
│   ├── retriever.py             # Coaching advice retriever
│   └── evaluation.py            # Eval harness (Hit Rate@3 target >= 0.82)
├── analytics/
│   ├── __init__.py
│   ├── models.py                # PostSessionReport JSON schema (Pydantic)
│   ├── posture_analytics.py     # PyTorch/NumPy posture stability processor
│   ├── speech_analytics.py      # Cadence (WPM) & filler word frequency
│   └── session_aggregator.py    # Aggregates time-series into structured report
├── api/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application factory
│   ├── routes.py                # REST endpoints for session lifecycle
│   └── websocket.py             # WebSocket streaming endpoint for video/audio & telemetry
├── docs/
│   ├── architecture.md          # Comprehensive system design, subsystem specs, ASCII diagrams
│   └── api_spec.md              # REST endpoints & WebSocket event payloads
├── reports/
│   └── sample_session_report.json # Reference post-session output JSON
└── tests/
    ├── __init__.py
    ├── test_video_pipeline.py   # Unit tests for capture, drop-stale, posture math & SLA
    ├── test_rag_engine.py       # RAG chunking & FAISS index tests
    └── test_analytics.py        # Posture stability & speech cadence tests
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Video Pipeline Unit Tests
```bash
python -m pytest tests/test_video_pipeline.py -v
```

### 3. Read Architecture & API Documentation
* [docs/architecture.md](docs/architecture.md): Subsystem specifications, per-frame latency budget breakdown, and ASCII data-flow diagrams.
* [docs/api_spec.md](docs/api_spec.md): Complete REST endpoint specs and WebSocket JSON/binary payload contracts.
