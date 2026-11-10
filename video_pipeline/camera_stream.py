"""Threaded video capture worker with bounded ring-buffer queue and stale-frame dropping."""

import logging
import queue
import threading
import time
from typing import Optional, Union
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .types import FramePacket

logger = logging.getLogger("presentation_ai.video_pipeline.camera_stream")


class FrameCaptureWorker:
    """Dedicated background thread for ingesting video frames without stalling inference."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        queue_maxsize: int = 2,
        drop_stale: bool = True
    ):
        self.source = source
        self.queue_maxsize = queue_maxsize
        self.drop_stale = drop_stale

        self._frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cap = None
        self._seq_counter = 0
        self._total_captured = 0
        self._dropped_frames = 0

    @property
    def frame_queue(self) -> queue.Queue[FramePacket]:
        return self._frame_queue

    @property
    def total_captured(self) -> int:
        return self._total_captured

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self) -> None:
        """Starts the capture worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Capture worker thread is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="FrameCaptureWorkerThread",
            daemon=True
        )
        self._thread.start()
        logger.info(f"FrameCaptureWorker started on source: {self.source}")

    def stop(self, timeout: float = 2.0) -> None:
        """Stops the capture worker thread gracefully."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._release_source()
        logger.info("FrameCaptureWorker stopped.")

    def push_external_frame(self, frame: np.ndarray) -> bool:
        """Allows programmatic push (e.g., from WebSocket ingestion) into the bounded buffer."""
        self._seq_counter += 1
        pkt = FramePacket(
            sequence_id=self._seq_counter,
            timestamp=time.perf_counter(),
            frame=frame
        )
        return self._dispatch_packet(pkt)

    def _dispatch_packet(self, packet: FramePacket) -> bool:
        """Pushes packet into bounded queue, dropping stale frames if queue is saturated."""
        self._total_captured += 1
        try:
            self._frame_queue.put_nowait(packet)
            return True
        except queue.Full:
            if self.drop_stale:
                try:
                    # Discard oldest frame to keep pipeline fresh
                    _ = self._frame_queue.get_nowait()
                    self._dropped_frames += 1
                    self._frame_queue.put_nowait(packet)
                    return True
                except (queue.Empty, queue.Full):
                    pass
            return False

    def _open_source(self) -> bool:
        """Initializes OpenCV VideoCapture."""
        if cv2 is None:
            logger.error("OpenCV (cv2) is not installed in the environment.")
            return False

        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            # Request low buffer size to reduce hardware latency if supported
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        except Exception as e:
            logger.exception(f"Error opening video source {self.source}: {e}")
            return False

    def _release_source(self) -> None:
        """Releases the camera handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.error(f"Error releasing video capture source: {e}")
            finally:
                self._cap = None

    def _capture_loop(self) -> None:
        """Continuous capture loop in dedicated OS thread."""
        if isinstance(self.source, (int, str)) and self.source != "EXTERNAL":
            if not self._open_source():
                return

        while not self._stop_event.is_set():
            if self._cap is None:
                # External feed mode - pause slightly
                time.sleep(0.005)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed or video source reached EOF.")
                time.sleep(0.01)
                continue

            self._seq_counter += 1
            pkt = FramePacket(
                sequence_id=self._seq_counter,
                timestamp=time.perf_counter(),
                frame=frame
            )
            self._dispatch_packet(pkt)
