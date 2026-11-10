"""FastAPI REST routes for session lifecycles, report retrieval, and health checks."""

import time
from typing import Dict, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from .session_manager import SessionManager, SessionState
from analytics.models import PostSessionReport

router = APIRouter(prefix="/api/v1", tags=["Sessions"])

# Module-level session manager reference (injected during app startup)
_session_manager: Optional[SessionManager] = None


def set_global_session_manager(manager: SessionManager) -> None:
    global _session_manager
    _session_manager = manager


def get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SessionManager is not initialized."
        )
    return _session_manager


# Request & Response Models
class SessionStartRequest(BaseModel):
    user_id: str = Field(default="usr_default")
    presentation_title: str = Field(default="Keynote Presentation")
    target_duration_minutes: Optional[int] = Field(default=15)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionStartResponse(BaseModel):
    session_id: str
    status: str
    created_at: float
    websocket_url: str
    sla_config: Dict[str, Any]


class SessionStopResponse(BaseModel):
    session_id: str
    status: str
    total_frames_processed: int
    duration_seconds: float
    report_url: str


@router.post("/sessions/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    req: SessionStartRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """Initializes a new presentation coaching session."""
    target_fps = req.config.get("target_fps", 30.0) if req.config else 30.0
    sla_ms = req.config.get("max_latency_sla_ms", 22.0) if req.config else 22.0

    state: SessionState = manager.create_session(
        user_id=req.user_id,
        presentation_title=req.presentation_title,
        target_fps=target_fps,
        max_latency_sla_ms=sla_ms
    )

    return SessionStartResponse(
        session_id=state.session_id,
        status=state.status,
        created_at=state.created_at,
        websocket_url=f"/ws/v1/session/{state.session_id}",
        sla_config={
            "target_fps": target_fps,
            "max_latency_sla_ms": sla_ms
        }
    )


@router.post("/sessions/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """Halts active streaming and generates the post-session report."""
    state = manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")

    report = manager.stop_session(session_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to synthesize session report.")

    frames_processed = state.pipeline.total_processed if state.pipeline else 0
    return SessionStopResponse(
        session_id=session_id,
        status="COMPLETED",
        total_frames_processed=frames_processed,
        duration_seconds=report.summary.duration_seconds,
        report_url=f"/api/v1/sessions/{session_id}/report"
    )


@router.get("/sessions/{session_id}/report", response_model=PostSessionReport)
async def get_session_report(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """Retrieves the completed structured evaluation report."""
    state = manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")

    if state.report is None:
        if state.status == "ACTIVE":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Session is still active. Please call stop endpoint first."
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not generated.")

    return state.report


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """Live telemetry status for an ongoing presentation."""
    state = manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")

    with state.lock:
        uptime = time.perf_counter() - state.start_time
        fps = state.pipeline.current_fps if state.pipeline else 0.0
        frames = state.pipeline.total_processed if state.pipeline else 0
        violations = state.pipeline.sla_violations if state.pipeline else 0

    return {
        "session_id": session_id,
        "status": state.status,
        "uptime_seconds": round(uptime, 2),
        "pipeline_fps": fps,
        "total_frames_processed": frames,
        "sla_violations": violations,
        "telemetry_points_count": len(state.telemetry_history)
    }


@router.get("/health")
async def health_check(manager: SessionManager = Depends(get_session_manager)):
    """System health and subsystem telemetry."""
    rag_indexed = manager.retriever.indexer.total_indexed if manager.retriever else 0
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "components": {
            "video_pipeline": {"status": "READY", "target_fps": 30.0, "latency_sla_ms": 22.0},
            "rag_engine": {"status": "READY", "total_chunks_indexed": rag_indexed},
            "analytics_engine": {"status": "READY"}
        }
    }


class GenerateQuestionsRequest(BaseModel):
    question_count: int = Field(default=3, ge=1, le=10)
    difficulty: Literal["foundational", "probing", "challenging"] = "probing"
    focus_area: Optional[str] = Field(default=None, description="Specific topic to drill questions on")


@router.post("/sessions/{session_id}/generate-questions")
async def generate_questions_route(
    session_id: str,
    req: Optional[GenerateQuestionsRequest] = None,
    manager: SessionManager = Depends(get_session_manager)
):
    """Analyzes session transcript and returns dynamic Q&A practice questions."""
    req = req or GenerateQuestionsRequest()
    result = await manager.generate_questions_for_session(
        session_id=session_id,
        question_count=req.question_count,
        difficulty=req.difficulty,
        focus_area=req.focus_area
    )
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found."
        )
    return result.to_dict()
