# Presentation AI: API & WebSocket Protocol Specification

## 1. Overview & Transport Protocol

The **Presentation AI** backend exposes dual communication interfaces:
1. **REST API (HTTP/1.1 & HTTP/2):** Handles session lifecycles, configuration provisioning, health checks, and post-session report retrieval.
2. **WebSocket Interface (`/ws/v1/session/{session_id}`):** Provides ultra-low-latency, full-duplex streaming for real-time video frames, audio chunks, pose telemetry, and coaching alerts.

* **Base URL:** `http://localhost:8000`
* **WebSocket URL:** `ws://localhost:8000/ws/v1/session/{session_id}`
* **Default Content-Type:** `application/json`

---

## 2. Session State Machine

```
              +---------------+
              |  INITIALIZING |
              +---------------+
                      |
           POST /sessions/start
                      v
              +---------------+
              |    ACTIVE     | <---+
              +---------------+     | Streaming frames
                      |             | & audio
          WS Connect / Stream       |
                      v             |
              +---------------+     |
              |   STREAMING   | ----+
              +---------------+
                      |
           POST /sessions/{id}/stop
                      v
              +---------------+
              |   ANALYZING   | (PyTorch + FAISS RAG)
              +---------------+
                      |
                      v
              +---------------+
              |   COMPLETED   | (Report Ready)
              +---------------+
```

---

## 3. REST API Endpoints

### 3.1. Initialize Session
* **Endpoint:** `POST /api/v1/sessions/start`
* **Description:** Initializes a new presentation coaching session and allocates dedicated queue buffers.

#### Request Headers
```http
Content-Type: application/json
X-Client-Version: 1.0.0
```

#### Request Payload
```json
{
  "user_id": "usr_94821a",
  "presentation_title": "Q4 Enterprise Strategy Keynote",
  "target_duration_minutes": 15,
  "config": {
    "target_fps": 30,
    "max_latency_sla_ms": 22.0,
    "speech_wpm_target_range": [130, 160],
    "posture_sensitivity": "normal",
    "enable_rag_coaching": true
  }
}
```

#### Response Payload (`201 Created`)
```json
{
  "session_id": "sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d",
  "status": "ACTIVE",
  "created_at": "2026-11-10T10:00:00Z",
  "websocket_url": "ws://localhost:8000/ws/v1/session/sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d",
  "sla_config": {
    "target_fps": 30,
    "max_latency_sla_ms": 22.0
  }
}
```

---

### 3.2. Terminate Session & Trigger Analytics
* **Endpoint:** `POST /api/v1/sessions/{session_id}/stop`
* **Description:** Stops active video/audio ingestion, gracefully flushes the capture queues, and triggers PyTorch post-session analytics and FAISS RAG enrichment.

#### Response Payload (`202 Accepted`)
```json
{
  "session_id": "sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d",
  "status": "ANALYZING",
  "total_frames_processed": 5400,
  "session_duration_seconds": 180.25,
  "average_fps": 29.96,
  "report_url": "/api/v1/sessions/sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d/report"
}
```

---

### 3.3. Retrieve Post-Session Evaluation Report
* **Endpoint:** `GET /api/v1/sessions/{session_id}/report`
* **Description:** Retrieves the structured JSON evaluation report synthesized by the PyTorch analytics engine and enriched with FAISS RAG coaching recommendations.

#### Response Payload (`200 OK`)
```json
{
  "session_id": "sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d",
  "status": "COMPLETED",
  "summary": {
    "overall_score": 84.5,
    "duration_seconds": 180.25,
    "total_words_spoken": 442,
    "average_cadence_wpm": 147.1,
    "filler_word_count": 8,
    "filler_word_ratio_pct": 1.81,
    "posture_stability_score": 88.2
  },
  "radar_dimensions": {
    "posture": 88.2,
    "pacing": 92.0,
    "gesture_expressiveness": 79.5,
    "verbal_clarity": 84.0,
    "energy_consistency": 81.3
  },
  "time_series_metrics": {
    "timestamps": [0.0, 15.0, 30.0, 45.0, 60.0],
    "wpm_rolling": [140.0, 155.0, 162.0, 144.0, 142.0],
    "posture_stability_rolling": [92.0, 89.5, 76.0, 88.0, 91.0],
    "hand_fidget_energy": [0.012, 0.015, 0.045, 0.018, 0.011]
  },
  "detected_defects": [
    {
      "defect_type": "SLUMPED_POSTURE",
      "timestamp_range": [32.0, 48.0],
      "severity": "MODERATE",
      "metric_value": "Spine tilt deviation exceeded 14.2 degrees"
    },
    {
      "defect_type": "FILLER_WORD_CLUSTER",
      "timestamp_range": [75.0, 88.0],
      "severity": "LOW",
      "metric_value": "4 occurrences of 'like' and 'um' within 13 seconds"
    }
  ],
  "rag_coaching_recommendations": [
    {
      "source_transcript_id": "transcript_chunk_412",
      "similarity_score": 0.894,
      "topic": "Upper Torso Alignment & Confidence Projection",
      "recommendation": "When moving between slides, anchor your feet shoulder-width apart and roll your shoulders back. Expanding your thoracic cage directly improves diaphragm breath support."
    },
    {
      "source_transcript_id": "transcript_chunk_108",
      "similarity_score": 0.867,
      "topic": "Cadence Modulation & Deliberate Pausing",
      "recommendation": "Replace verbal fillers with a silent two-second pause before transitioning to financial data points. This creates gravitas and lets your audience process key numbers."
    }
  ]
}
```

---

### 3.4. System Health & Readiness
* **Endpoint:** `GET /api/v1/health`
* **Description:** Returns hardware utilization, model engine status, and FAISS vector index status.

#### Response Payload (`200 OK`)
```json
{
  "status": "HEALTHY",
  "timestamp": "2026-11-10T10:05:00Z",
  "components": {
    "mediapipe_pose": {
      "status": "READY",
      "model_complexity": 1,
      "avg_inference_latency_ms": 13.8
    },
    "faiss_rag": {
      "status": "INDEXED",
      "total_transcripts": 1000,
      "total_chunks": 4210,
      "index_type": "IndexHNSWFlat",
      "hnsw_m": 16,
      "hit_rate_at_3": 0.835
    },
    "pytorch_analytics": {
      "status": "READY",
      "device": "cpu"
    }
  }
}
```

---

## 4. WebSocket Streaming Protocol (`/ws/v1/session/{session_id}`)

### 4.1. Handshake & Initialization

```
Client                                             Server
  |                                                  |
  | -------- 1. HTTP Upgrade (GET /ws/v1/...) -----> |
  | <------- 2. 101 Switching Protocols ------------ |
  | <------- 3. Event: connection_ack -------------- |
  |                                                  |
  | ======== Full-Duplex Low-Latency Stream ======== |
```

#### Event: `connection_ack` (Server -> Client)
```json
{
  "event": "connection_ack",
  "session_id": "sess_883f9b20-1a42-4b71-b0e2-8874fa19e03d",
  "timestamp": 1731213600.120,
  "config": {
    "target_fps": 30,
    "max_frame_bytes": 1048576,
    "expected_audio_format": "audio/pcm;rate=16000;channels=1"
  }
}
```

---

### 4.2. Inbound Client Messages (Client -> Server)

#### Message A: `frame_packet` (Video Frame)
* **Payload Structure:**
```json
{
  "type": "frame_packet",
  "seq_id": 1420,
  "timestamp": 1731213605.421,
  "format": "jpeg",
  "data": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP... [Base64 Encoded Image Data]"
}
```

* **Binary Alternative:** Raw JPEG byte buffer with a 16-byte fixed header:
  * Bytes 0–7: `uint64` Sequence Number (big-endian)
  * Bytes 8–15: `float64` Timestamp (epoch seconds)
  * Bytes 16+: Raw JPEG/PNG image binary payload.

#### Message B: `audio_packet` (Audio PCM Chunk)
```json
{
  "type": "audio_packet",
  "seq_id": 284,
  "timestamp": 1731213605.500,
  "sample_rate": 16000,
  "channels": 1,
  "data": "5PDw8PDw8PDw8PDw8PD... [Base64 Encoded 16-bit PCM Linear]"
}
```

#### Message C: `client_heartbeat`
```json
{
  "type": "client_heartbeat",
  "timestamp": 1731213610.000
}
```

---

### 4.3. Outbound Server Messages (Server -> Client)

#### Message A: `live_metrics` (Per-Frame / Periodic Telemetry)
* **Frequency:** Emitted every frame or decimated to 15 Hz for network efficiency.
* **Processing Guarantee:** Latency $< 22\text{ms}$.
```json
{
  "event": "live_metrics",
  "seq_id": 1420,
  "timestamp": 1731213605.438,
  "latency": {
    "capture_ms": 2.8,
    "inference_ms": 13.4,
    "posture_calc_ms": 1.1,
    "total_frame_latency_ms": 17.3,
    "sla_satisfied": true
  },
  "pipeline_fps": 30.1,
  "landmarks": {
    "left_shoulder": {"x": 0.382, "y": 0.412, "z": -0.154, "visibility": 0.98},
    "right_shoulder": {"x": 0.618, "y": 0.415, "z": -0.149, "visibility": 0.97},
    "spine_midpoint": {"x": 0.500, "y": 0.560, "z": -0.110, "visibility": 0.95},
    "left_wrist": {"x": 0.320, "y": 0.680, "z": -0.050, "visibility": 0.92},
    "right_wrist": {"x": 0.690, "y": 0.675, "z": -0.045, "visibility": 0.91}
  },
  "posture": {
    "shoulder_tilt_degrees": 0.42,
    "spine_deviation_degrees": 2.15,
    "hand_movement_velocity": 0.014,
    "current_stability_index": 94.6
  },
  "speech": {
    "instantaneous_wpm": 145.0,
    "filler_word_detected": null
  }
}
```

#### Message B: `coaching_alert` (Real-Time HUD Warning)
* **Trigger:** Dispatched immediately when a body language defect or speech anomaly persists across a rolling window ($\ge 1.5\text{s}$).
```json
{
  "event": "coaching_alert",
  "alert_id": "alt_0921a",
  "timestamp": 1731213607.210,
  "level": "WARNING",
  "category": "BODY_LANGUAGE",
  "code": "SLUMPED_SPINE_DETECTED",
  "message": "Spine plumb line is leaning forward 15°. Straighten up and align shoulders over hips.",
  "duration_active_ms": 1800,
  "metric_trigger": {
    "current_value": 15.2,
    "threshold": 10.0
  }
}
```

#### Message C: `error_event`
```json
{
  "event": "error_event",
  "code": "FRAME_DROP_EXCEEDED",
  "message": "Client delivery rate dropped below 20 FPS. Throttling inference quality.",
  "timestamp": 1731213612.000
}
```

---

## 5. HTTP & WebSocket Error Codes

| Status / Code | Identifier | Description | Action Required |
| :--- | :--- | :--- | :--- |
| `400 Bad Request` | `INVALID_PAYLOAD` | Missing required parameters or malformed JSON | Correct schema |
| `404 Not Found` | `SESSION_NOT_FOUND` | Specified `session_id` does not exist | Create session first |
| `422 Unprocessable` | `INVALID_VIDEO_FORMAT` | Frame payload cannot be decoded via OpenCV | Check JPEG encoding |
| `WS 1008 Policy` | `SLA_VIOLATION_FATAL` | Consecutive buffer overflow | Reduce frame resolution |
| `WS 1011 Server Error`| `MODEL_INFERENCE_FAIL` | MediaPipe / OpenCV C++ runtime exception | Restart worker thread |
| `503 Service Unavail`| `FAISS_INDEX_UNLOADED` | RAG vector index not loaded in memory | Build/load index |
