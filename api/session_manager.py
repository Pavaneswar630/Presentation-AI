"""Thread-safe session state manager and pipeline lifecycle orchestrator."""

import logging
import threading
import time
import uuid
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from video_pipeline.pose_worker import DualThreadedPosePipeline
from video_pipeline.types import PoseFrameResult
from analytics.models import TelemetryPoint, PostSessionReport
from analytics.speech_analytics import SpokenWordToken
from analytics.session_aggregator import SessionDataAggregator
from rag_engine.retriever import CoachingRetriever
from rag_engine.question_generator import (
    LLMQuestionGenerator,
    QuestionGenerationResult,
    LiveProbePrompt
)

logger = logging.getLogger("presentation_ai.api.session_manager")


@dataclass
class SessionState:
    """Live state for an active coaching session."""
    session_id: str
    user_id: str
    title: str
    status: str = "ACTIVE"
    created_at: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.perf_counter)
    stopped_at: Optional[float] = None
    pipeline: Optional[DualThreadedPosePipeline] = None
    telemetry_history: List[TelemetryPoint] = field(default_factory=list)
    spoken_tokens: List[SpokenWordToken] = field(default_factory=list)
    report: Optional[PostSessionReport] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    """Manages active coaching sessions, telemetry accumulation, and report compilation."""

    def __init__(self, retriever: Optional[CoachingRetriever] = None):
        self.retriever = retriever
        self.aggregator = SessionDataAggregator(retriever=retriever)
        indexer = retriever.indexer if retriever else None
        self.question_generator = LLMQuestionGenerator(indexer=indexer, retriever=retriever)
        self._sessions: Dict[str, SessionState] = {}
        self._global_lock = threading.Lock()

    def create_session(
        self,
        user_id: str = "anonymous",
        presentation_title: str = "Untitled Keynote",
        target_fps: float = 30.0,
        max_latency_sla_ms: float = 22.0,
        session_id: Optional[str] = None
    ) -> SessionState:
        """Initializes a new session and launches an external-source video pipeline."""
        session_id = session_id or f"sess_{uuid.uuid4()}"
        pipeline = DualThreadedPosePipeline(
            video_source="EXTERNAL",
            max_latency_sla_ms=max_latency_sla_ms,
            drop_stale=True
        )

        state = SessionState(
            session_id=session_id,
            user_id=user_id,
            title=presentation_title,
            pipeline=pipeline
        )

        # Connect pipeline callback to accumulate session telemetry
        def on_pose_result(result: PoseFrameResult):
            with state.lock:
                pt = TelemetryPoint(
                    timestamp=time.perf_counter() - state.start_time,
                    shoulder_tilt_degrees=result.posture.shoulder_tilt_degrees,
                    spine_deviation_degrees=result.posture.spine_deviation_degrees,
                    hand_movement_velocity=result.posture.hand_movement_velocity,
                    stability_index=result.posture.stability_index,
                    is_slumping=result.posture.is_slumping,
                    are_shoulders_uneven=result.posture.are_shoulders_uneven,
                    is_fidgeting=result.posture.is_fidgeting
                )
                state.telemetry_history.append(pt)

        pipeline.register_callback(on_pose_result)
        pipeline.start()

        with self._global_lock:
            if session_id in self._sessions:
                pipeline.stop()
                raise ValueError(f"Session {session_id} already exists.")
            self._sessions[session_id] = state

        logger.info(f"Initialized coaching session {session_id} for user {user_id}")
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieves session state by ID."""
        with self._global_lock:
            return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> Optional[PostSessionReport]:
        """Stops the session pipeline and synthesizes the post-session evaluation report."""
        state = self.get_session(session_id)
        if not state:
            return None

        with state.lock:
            if state.status == "COMPLETED" and state.report is not None:
                return state.report

            state.status = "ANALYZING"
            state.stopped_at = time.time()
            duration_seconds = max(0.1, time.perf_counter() - state.start_time)

            # Stop pipeline worker
            if state.pipeline is not None:
                state.pipeline.stop()

            # Compile report
            report = self.aggregator.generate_report(
                session_id=state.session_id,
                duration_seconds=duration_seconds,
                telemetry=state.telemetry_history,
                spoken_tokens=state.spoken_tokens
            )
            state.report = report
            state.status = "COMPLETED"

        logger.info(f"Session {session_id} completed. Overall score: {report.summary.overall_score}")
        return report

    def add_speech_tokens(self, session_id: str, tokens: List[SpokenWordToken]) -> bool:
        """Appends streaming STT word tokens to the session state."""
        state = self.get_session(session_id)
        if not state:
            return False

        with state.lock:
            state.spoken_tokens.extend(tokens)
        return True

    async def generate_questions_for_session(
        self,
        session_id: str,
        question_count: int = 3,
        difficulty: str = "probing",
        focus_area: Optional[str] = None
    ) -> Optional[QuestionGenerationResult]:
        """Analyzes session transcript and generates targeted audience questions via RAG + LLM."""
        state = self.get_session(session_id)
        if not state:
            return None

        with state.lock:
            if state.spoken_tokens:
                transcript_text = " ".join([t.word for t in state.spoken_tokens])
            else:
                transcript_text = f"Keynote presentation on {state.title}."

        result = await self.question_generator.generate_questions(
            session_id=session_id,
            transcript_text=transcript_text,
            question_count=question_count,
            difficulty=difficulty,
            focus_area=focus_area
        )
        return result

    def generate_live_probe_for_session(self, session_id: str) -> Optional[LiveProbePrompt]:
        """Generates an instantaneous live practice probe based on recent speech."""
        state = self.get_session(session_id)
        if not state:
            return None

        with state.lock:
            cutoff = max(0.0, time.perf_counter() - state.start_time - 30.0)
            recent_tokens = [
                token for token in state.spoken_tokens
                if token.end_time >= cutoff
            ]
            if recent_tokens:
                recent_text = " ".join([t.word for t in recent_tokens])
            else:
                recent_text = f"Presentation topic: {state.title}."

        return self.question_generator.generate_live_probe(recent_text)
