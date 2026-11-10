"""FastAPI application factory and lifecycle orchestrator for Presentation AI."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, set_global_session_manager
from .websocket import ws_router
from .session_manager import SessionManager
from rag_engine.corpus_generator import TranscriptCorpusGenerator
from rag_engine.chunker import TranscriptChunker
from rag_engine.indexer import FAISSHNSWIndexer
from rag_engine.retriever import CoachingRetriever

logger = logging.getLogger("presentation_ai.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes shared RAG knowledge base and session manager during startup."""
    logger.info("Initializing Presentation AI backend subsystems...")

    # 1. Initialize RAG Engine with 1,000 Coaching Transcripts
    corpus, _ = TranscriptCorpusGenerator.generate_corpus(count=1000)
    chunker = TranscriptChunker(chunk_size=256, chunk_overlap=32)
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)

    chunks = chunker.chunk_corpus(corpus)
    indexer.add_chunks(chunks)

    retriever = CoachingRetriever(indexer=indexer)

    # 2. Initialize SessionManager and inject globally into API routes
    session_manager = SessionManager(retriever=retriever)
    set_global_session_manager(session_manager)
    app.state.session_manager = session_manager

    logger.info("Presentation AI backend ready to serve API and WebSocket requests.")
    yield

    logger.info("Shutting down Presentation AI backend...")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI application."""
    import os
    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles

    app = FastAPI(
        title="Presentation AI Coaching API",
        version="1.0.0",
        description="Real-time speech cadence and body language coaching system with OpenCV/MediaPipe, FAISS HNSW, and PyTorch analytics.",
        lifespan=lifespan
    )

    # Enable CORS for frontend clients (web HUD, dashboard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Mount REST endpoints and WebSocket stream
    app.include_router(router)
    app.include_router(ws_router)

    # Mount static frontend
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    return app


app = create_app()
