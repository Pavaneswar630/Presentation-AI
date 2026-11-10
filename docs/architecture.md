# Presentation AI: System Architecture Specification

## 1. Executive Summary & Architectural Goals

**Presentation AI** is an enterprise-grade, real-time coaching platform that evaluates speaking cadence, speech clarity, filler word frequency, and physical body language (posture, spinal alignment, gesture stability) during live presentations.

### Key Performance Indicators & SLA Targets
* **Frame Throughput:** Steady **30 FPS** processing over continuous video streams.
* **Per-Frame Processing Latency:** **< 22ms** per frame end-to-end (Capture -> MediaPipe Pose Inference -> Geometric Posture Extraction -> WebSocket Broadcast).
* **RAG Retrieval Quality:** Offline FAISS HNSW index ($M=16$) achieving **Hit Rate@3 >= 0.82** across 1,000 reference coaching transcripts.
* **Concurrency & Stability:** Zero frame accumulation / pipeline lag via bounded ring-buffers and stale-frame drop mechanisms.

---

## 2. High-Level Subsystem Architecture

The system comprises four tightly decoupled subsystems:

```
+---------------------------------------------------------------------------------------------------+
|                                      PRESENTATION AI CORE                                         |
+---------------------------------------------------------------------------------------------------+
                                                  |
         +----------------------------------------+----------------------------------------+
         |                                        |                                        |
         v                                        v                                        v
+------------------+                    +--------------------+                   +------------------+
|  VIDEO PIPELINE  |                    |     RAG ENGINE     |                   | ANALYTICS ENGINE |
| (OpenCV + MPose) |                    | (FAISS HNSW, M=16) |                   | (PyTorch + NumPy)|
+------------------+                    +--------------------+                   +------------------+
         |                                        |                                        |
         +----------------------------------------+----------------------------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      |   FASTAPI & WEBSOCKET |
                                      | (Real-Time Streaming) |
                                      +-----------------------+
```

---

## 3. Subsystem Specifications

### 3.1. Dual-Threaded Video & Pose Pipeline (`video_pipeline/`)

To guarantee a steady 30 FPS under heavy computer vision workloads without blocking network I/O or dropping camera capture buffers, the video pipeline utilizes a **Producer-Consumer Multi-Threaded Architecture**.

```
                           DUAL-THREADED VIDEO PIPELINE
                           
 [ Camera / RTSP / WS ]
          |
          v
  +---------------+        Non-blocking put()        +------------------------+
  | CaptureWorker | -------------------------------> | Ring Buffer / Queue    |
  |   (Thread 1)  |   (Drops stale frame if full)    | (maxsize=2, drop_old)  |
  +---------------+                                  +------------------------+
                                                                  |
                                                           get_nowait()
                                                                  v
                                                     +------------------------+
                                                     | PoseInferenceWorker    |
                                                     |      (Thread 2)        |
                                                     |  MediaPipe Pose (0/1)  |
                                                     +------------------------+
                                                                  |
                                                                  v
                                                     +------------------------+
                                                     | Posture Calculator     |
                                                     | (Spine, Shoulder, Hand)|
                                                     +------------------------+
                                                                  |
                                                                  v
                                                     [ Real-Time Telemetry ]
```

#### Thread 1: `FrameCaptureWorker`
* **Responsibility:** Ingests raw video frames from `cv2.VideoCapture` or decoded WebSocket buffers.
* **Buffer Mechanism:** Bounded `queue.Queue(maxsize=2)` or atomic ring buffer. If the queue is saturated, the worker pops the stale frame and immediately inserts the newest frame, preventing latency build-up.
* **Target Latency:** 2.5ms - 4.0ms per capture.

#### Thread 2: `PoseInferenceWorker`
* **Responsibility:** Ingests the freshest RGB frame, transforms color spaces (`cv2.cvtColor(BGR2RGB)`), and feeds it into the MediaPipe Pose estimator.
* **Model Parameters:**
  * `model_complexity=1` (Balanced for real-time accuracy and <16ms inference on modern CPUs/GPUs).
  * `smooth_landmarks=True` (Applies 1-Euro filtering internally to minimize jitter).
  * `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.
* **Extracted Keypoints (MediaPipe 33-landmark schema):**
  * **Shoulders:** Left Shoulder (idx 11), Right Shoulder (idx 12).
  * **Spine / Torso:** Midpoint of Shoulders $P_{mid\_shoulder} = (P_{11} + P_{12}) / 2$ and Midpoint of Hips $P_{mid\_hip} = (P_{23} + P_{24}) / 2$.
  * **Hands & Wrists:** Left Wrist (15), Right Wrist (16), Left Elbow (13), Right Elbow (14).
  * **Head / Orientation:** Nose (0), Left Ear (7), Right Ear (8).

#### Per-Frame Processing Latency Budget (< 22ms Total)
| Pipeline Stage | Allocated Latency |
| :--- | :--- |
| Frame Capture / Decompression | 3.0 ms |
| Color Space Conversion (BGR -> RGB) | 0.8 ms |
| MediaPipe Pose Forward Pass | 14.5 ms |
| Posture Trigonometry & Feature Extraction | 1.2 ms |
| Thread Synchronization & Queue Dispatch | 0.5 ms |
| Buffer Overhead Margin | 1.5 ms |
| **Total Per-Frame Latency** | **21.5 ms < 22.0 ms SLA** |

---

### 3.2. Offline & Real-Time RAG Subsystem (`rag_engine/`)

The RAG engine acts as an expert presentation coach knowledge base. It indexes high-quality coaching transcripts, TED talk evaluations, public speaking rubrics, and diagnostic tips.

```
                            OFFLINE RAG INGESTION PIPELINE
                            
 [ 1,000 Transcripts ]
          |
          v
  +---------------+      Chunk Size = 256 tokens       +------------------------+
  |  Text Chunker | --------------------------------> | Chunks + Metadata      |
  | (Token-Aware) |      Overlap    = 32 tokens       | (Speaker, Topic, Time) |
  +---------------+                                   +------------------------+
                                                                  |
                                                                  v
                                                      +------------------------+
                                                      | BGE-Large-En Embedder  |
                                                      | (1024-dim, Normalized) |
                                                      +------------------------+
                                                                  |
                                                                  v
                                                      +------------------------+
                                                      | FAISS HNSW Index       |
                                                      | (M=16, efConst=64)     |
                                                      +------------------------+
```

#### Ingestion & Tokenizer Configuration
* **Corpus:** 1,000 domain-specific speech coaching transcripts.
* **Chunking Strategy:** Sliding window chunker configured with:
  * Chunk size: 256 tokens
  * Chunk overlap: 32 tokens
  * Metadata injection: Session domain, topic, severity flag, speaker archetype.
* **Embeddings:** `BAAI/bge-large-en-v1.5`
  * Dimension $d = 1024$.
  * Embeddings are $L_2$-normalized prior to indexing to permit cosine distance calculations via dot products.

#### Vector Store: FAISS HNSW
* **Index Configuration:** `IndexHNSWFlat(d=1024, M=16, metric=METRIC_INNER_PRODUCT)`
* **Graph Parameters:**
  * $M = 16$: Number of bidirectional links per vector node.
  * $efConstruction = 64$: Explored neighbor list during graph construction.
  * $efSearch = 32$: Search-time exploration depth balancing recall and sub-millisecond retrieval.
* **Evaluation Target:** Tested against 1,000 annotated queries targeting **Hit Rate@3 >= 0.82** and **MRR@3 >= 0.70**.

---

### 3.3. Post-Session Analytics & PyTorch Engine Subsystem (`analytics/`)

When a coaching session concludes, raw time-series metrics accumulated across hundreds or thousands of frames are ingested into the PyTorch/NumPy post-session analytics engine.

```
                         POST-SESSION ANALYTICS ENGINE
                         
 [ Raw Telemetry Stream (Posture, Audio Tokens, Timestamps) ]
                             |
                             v
                 +-----------------------+
                 | Time-Series Tensorizer|
                 +-----------------------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+------------------+                   +--------------------+
| PyTorch Posture  |                   | Speech Analytics   |
| Stability Engine |                   | Cadence & Fillers  |
+------------------+                   +--------------------+
         |                                       |
         | * Spine Slump Tensor                  | * 15s Windowed WPM
         | * Shoulder Angle Variance             | * Filler Freq Ratio
         | * Hand Velocity Jitter                | * Pacing Deviation
         |                                       |
         +-------------------+-------------------+
                             |
                             v
               +---------------------------+
               | Multi-Dimensional Scoring |
               |     (0 - 100 Scales)      |
               +---------------------------+
                             |
                             v
               +---------------------------+
               |  RAG Coaching Enrichment  |
               +---------------------------+
                             |
                             v
               [ Structured JSON Report ]
```

#### Mathematical Formulation

1. **Shoulder Tilt Angle ($\theta_{shoulder}$):**
   $$\theta_{shoulder}(t) = \arctan2\left(y_{R\_shoulder}(t) - y_{L\_shoulder}(t), x_{R\_shoulder}(t) - x_{L\_shoulder}(t)\right)$$

2. **Spine Plumb Line Vector & Deviation ($\vec{V}_{spine}$):**
   $$\vec{P}_{mid\_shoulder} = \frac{\vec{P}_{11} + \vec{P}_{12}}{2}, \quad \vec{P}_{mid\_hip} = \frac{\vec{P}_{23} + \vec{P}_{24}}{2}$$
   $$\vec{V}_{spine}(t) = \vec{P}_{mid\_shoulder}(t) - \vec{P}_{mid\_hip}(t)$$
   $$\phi_{spine}(t) = \arccos\left(\frac{\vec{V}_{spine}(t) \cdot [0, -1]^T}{\|\vec{V}_{spine}(t)\|_2}\right)$$

3. **Hand Fidgeting & Kinetic Energy ($E_{hand}$):**
   $$v_{wrist}(t) = \frac{\|\vec{P}_{wrist}(t) - \vec{P}_{wrist}(t - \Delta t)\|_2}{\Delta t}$$
   $$E_{hand}(t) = \frac{1}{2}\left(v_{L\_wrist}^2(t) + v_{R\_wrist}^2(t)\right)$$

4. **PyTorch Vectorized Posture Stability Score ($S_{posture} \in [0, 100]$):**
   Given temporal tensors $\mathbf{\Theta} \in \mathbb{R}^T$ and $\mathbf{\Phi} \in \mathbb{R}^T$:
   $$\sigma^2_\theta = \mathrm{Var}(\mathbf{\Theta}), \quad \sigma^2_\phi = \mathrm{Var}(\mathbf{\Phi})$$
   $$S_{posture} = 100 \cdot \exp\left( - \left[ \alpha \sigma^2_\theta + \beta \sigma^2_\phi + \gamma \, \mathrm{Mean}(\mathbf{E}_{hand}) \right] \right)$$
   Where $\alpha=2.5, \beta=3.0, \gamma=1.2$ are calibrated normalization coefficients.

5. **Speaking Cadence & Filler Frequency:**
   $$\text{WPM}(w) = \frac{N_{words}(w)}{\Delta t_{minutes}(w)}$$
   $$\text{Filler Ratio} = \frac{N_{fillers}}{N_{total\_words}} \times 100\%$$
   $$\text{Optimal Cadence Penalty} = \max\left(0, 1 - \frac{|\text{WPM} - 145|}{45}\right) \times 100$$

---

## 4. Comprehensive ASCII Data-Flow Diagrams

### 4.1. Real-Time Streaming & Inference Data-Flow

```
+--------+                 +---------------+              +------------------+              +---------------+
| Client |                 |   FastAPI     |              |   FrameCapture   |              | PoseInference |
| (Web)  |                 | WebSocket Svr |              |      Worker      |              |    Worker     |
+--------+                 +---------------+              +------------------+              +---------------+
    |                              |                                |                               |
    | 1. WS Connect (/ws/{id})     |                                |                               |
    |----------------------------->|                                |                               |
    | 2. WS Handshake ACK          |                                |                               |
    |<-----------------------------|                                |                               |
    |                              |                                |                               |
    | 3. Send Frame (JPEG/Binary)  |                                |                               |
    |----------------------------->| 4. Push to Ring Buffer         |                               |
    |                              |------------------------------->|                               |
    |                              |                                | 5. Non-blocking Pop           |
    |                              |                                |------------------------------>|
    |                              |                                |                               | 6. MediaPipe Pose
    |                              |                                |                               |    Inference (<15ms)
    |                              |                                |                               | 7. Calc Stability
    |                              |                                |                               |    (<2ms)
    |                              | 8. Telemetry Metric Event      |                               |
    |                              |<---------------------------------------------------------------|
    | 9. WS Emit Metric JSON       |                                |                               |
    |<-----------------------------|                                |                               |
    |    (FPS, Angles, Alert)      |                                |                               |
    |                              |                                |                               |
    | 10. Send Audio Chunk (PCM)   |                                |                               |
    |----------------------------->| 11. Append to Audio Buffer     |                               |
    |                              |     (Rolling STT & WPM)        |                               |
    | 12. WS Emit Speech Alert     |                                |                               |
    |<-----------------------------|                                |                               |
```

### 4.2. End-to-End System Integration Flow

```
========================================================================================================================
                                           END-TO-END SYSTEM INTEGRATION
========================================================================================================================

 [ VIDEO SOURCE ]               [ AUDIO SOURCE ]
   (Webcam/WS)                    (Microphone/WS)
        |                                |
        | Raw Frames                     | PCM Chunks
        v                                v
+----------------------+       +------------------------+
| FrameCaptureWorker   |       | Audio Stream Ingestion |
| (RingBuffer maxsz=2) |       | (16kHz Mono Buffer)    |
+----------------------+       +------------------------+
        |                                |
        | Fresh RGB                      | Audio Window
        v                                v
+----------------------+       +------------------------+
| PoseInferenceWorker  |       | STT / Whisper Engine   |
| (MediaPipe Pose)     |       | (Transcript Stream)    |
+----------------------+       +------------------------+
        |                                |
        | 33 Landmarks                   | Words + Timestamps
        v                                v
+----------------------+       +------------------------+
| Posture Calculator   |       | Filler & Cadence Calc  |
| - Shoulder Tilt      |       | - Words Per Minute     |
| - Spine Alignment    |       | - Filler Word Tally    |
| - Hand Jerk Velocity |       | - Pacing Variation     |
+----------------------+       +------------------------+
        |                                |
        +---------------+----------------+
                        |
                        v
       +-----------------------------------+
       | Live WebSocket Feedback Dispatch |  --> (Client Real-Time HUD Alerts)
       | - Per-frame Latency & 30 FPS      |
       +-----------------------------------+
                        |
                        | (Accumulate Session Time-Series)
                        v
       +-----------------------------------+
       | Session Telemetry Buffer Store    |
       +-----------------------------------+
                        |
                        | User Triggers: POST /api/v1/sessions/{id}/stop
                        v
       +-----------------------------------+
       | PyTorch / NumPy Analytics Engine  |
       | - Posture Stability Curve (S)     |
       | - Hand Fidgeting FFT Energy       |
       | - Cadence Consistency Index       |
       +-----------------------------------+
                        |
                        | Defect Profile: ["slumped_posture", "excessive_fillers_um"]
                        v
       +-----------------------------------+
       | Offline FAISS HNSW RAG Engine     |
       | - Query: Defect semantic prompt   |
       | - Top-3 Coaching Recommendations  |
       +-----------------------------------+
                        |
                        v
       +-----------------------------------+
       | Final Structured JSON Report      |  --> (Client Session Dashboard)
       +-----------------------------------+
```

---

## 5. Concurrency, GIL Mitigation & Performance Optimization

1. **Python Global Interpreter Lock (GIL) Mitigation:**
   * OpenCV (`cv2`) and MediaPipe run their core compute in compiled C++ threads, releasing the GIL during tensor evaluation.
   * Frame capture runs in an isolated Python `threading.Thread` dedicated solely to socket/hardware device reading and `put_nowait()` dispatch.
   * PyTorch tensor operations leverage native vectorized multi-threaded BLAS/MKL kernels.
2. **Backpressure & Drop-Frame Guarantee:**
   * The frame queue uses a strict capacity of 2. If the inference worker is occupied, incoming frames overwrite the queue slot rather than buffering.
   * Total latency cannot drift or accumulate over long presentation sessions.
3. **Graceful Shutdown & Signal Handling:**
   * Thread workers monitor an atomic `threading.Event` stop flag.
   * Upon receipt of a termination signal or session stop command, camera handles and native contexts are closed safely within < 50ms.
