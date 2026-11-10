"""WebSocket router for bi-directional video/audio streaming and real-time coaching HUD telemetry."""

import asyncio
import base64
import json
import logging
import time
from typing import Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from .session_manager import SessionManager, SessionState
from .routes import get_session_manager
from video_pipeline.types import PoseFrameResult
from analytics.speech_analytics import SpokenWordToken

logger = logging.getLogger("presentation_ai.api.websocket")

ws_router = APIRouter(tags=["WebSocket Stream"])


def decode_image_frame(data: str | bytes) -> Optional[np.ndarray]:
    """Decodes Base64 string or raw bytes into a numpy image array."""
    try:
        if isinstance(data, str):
            # Strip data URL prefix if present
            if "," in data:
                data = data.split(",", 1)[1]
            raw_bytes = base64.b64decode(data)
        else:
            raw_bytes = data

        if cv2 is not None:
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        # Fallback 240x320 synthetic array if cv2 decode unavailable
        return np.zeros((240, 320, 3), dtype=np.uint8)
    except Exception as e:
        logger.warning(f"Failed to decode video frame: {e}")
        return None


@ws_router.websocket("/ws/v1/session/{session_id}")
async def presentation_stream(websocket: WebSocket, session_id: str):
    """Full-duplex WebSocket stream handling live frames and broadcasting <22ms pose telemetry."""
    await websocket.accept()

    try:
        manager: SessionManager = get_session_manager()
    except Exception:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="SessionManager not ready.")
        return

    state: SessionState = manager.get_session(session_id)
    if not state:
        # Create ad-hoc session if not pre-initialized
        state = manager.create_session(
            user_id="stream_user",
            presentation_title=f"Live Session {session_id}",
            session_id=session_id
        )

    # 1. Send Handshake ACK
    ack_payload = {
        "event": "connection_ack",
        "session_id": session_id,
        "timestamp": time.time(),
        "config": {
            "target_fps": 30.0,
            "max_latency_sla_ms": 22.0,
            "transport": "websocket-duplex"
        }
    }
    await websocket.send_text(json.dumps(ack_payload))

    loop = asyncio.get_running_loop()
    outbound_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10)

    # 2. Register callback on pipeline worker
    def on_pipeline_result(result: PoseFrameResult):
        # Format live telemetry event
        event = {
            "event": "live_metrics",
            "seq_id": result.sequence_id,
            "timestamp": result.timestamp,
            "pipeline_fps": round(result.pipeline_fps, 1),
            "latency": result.latency.to_dict(),
            "landmarks": result.landmarks.to_dict(),
            "posture": result.posture.to_dict()
        }

        # Check for immediate coaching alert triggers
        alerts = []
        if result.posture.is_slumping:
            alerts.append({
                "event": "coaching_alert",
                "level": "WARNING",
                "code": "SLUMPED_SPINE_DETECTED",
                "message": f"Spine deviation at {result.posture.spine_deviation_degrees:.1f}°. Roll shoulders back and align spine."
            })
        elif result.posture.are_shoulders_uneven:
            alerts.append({
                "event": "coaching_alert",
                "level": "WARNING",
                "code": "SHOULDER_ASYMMETRY",
                "message": f"Shoulder tilt at {result.posture.shoulder_tilt_degrees:.1f}°. Balance your stance evenly."
            })

        # Thread-safe dispatch to async WebSocket queue
        try:
            loop.call_soon_threadsafe(outbound_queue.put_nowait, event)
            for alert in alerts:
                loop.call_soon_threadsafe(outbound_queue.put_nowait, alert)
        except asyncio.QueueFull:
            pass  # Drop telemetry frame to prevent client backpressure lag

    if state.pipeline is not None:
        state.pipeline.register_callback(on_pipeline_result)

    # Background task to stream outbound events
    async def sender_task():
        try:
            while True:
                msg = await outbound_queue.get()
                await websocket.send_text(json.dumps(msg))
        except asyncio.CancelledError:
            pass
        except Exception as ex:
            logger.debug(f"Sender task exited: {ex}")

    sender = asyncio.create_task(sender_task())

    # Inbound message receiver loop
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "frame_packet":
                        frame_data = payload.get("data")
                        if frame_data:
                            img = decode_image_frame(frame_data)
                            if img is not None and state.pipeline is not None:
                                state.pipeline.push_external_frame(img)

                    elif msg_type == "audio_packet":
                        # Process spoken words / STT transcript tokens
                        word = payload.get("word")
                        if word:
                            tok = SpokenWordToken(
                                word=word,
                                start_time=payload.get("start_time", time.perf_counter() - state.start_time),
                                end_time=payload.get("end_time", time.perf_counter() - state.start_time + 0.3)
                            )
                            manager.add_speech_tokens(session_id, [tok])

                    elif msg_type == "client_heartbeat":
                        await websocket.send_text(json.dumps({"event": "server_pong", "timestamp": time.time()}))

                    elif msg_type == "request_live_probes":
                        probe = manager.generate_live_probe_for_session(session_id)
                        if probe:
                            await websocket.send_text(json.dumps({
                                "event": "live_probe_question",
                                "probe": probe.to_dict()
                            }))

                except json.JSONDecodeError:
                    pass

            elif "bytes" in message and message["bytes"]:
                # Binary frame buffer
                img = decode_image_frame(message["bytes"])
                if img is not None and state.pipeline is not None:
                    state.pipeline.push_external_frame(img)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from session {session_id}")
    finally:
        sender.cancel()
