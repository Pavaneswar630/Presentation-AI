"""FAISS HNSW (M=16) indexing engine with BGE-large-en embeddings."""

import json
import logging
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .chunker import TranscriptChunk

logger = logging.getLogger("presentation_ai.rag_engine.indexer")


class BGEEmbedder:
    """Generates 1024-dimensional normalized dense embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", dimension: int = 1024):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

        if SentenceTransformer is not None:
            try:
                logger.info(f"Loading SentenceTransformer embedding model: {model_name}")
                self._model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace model '{model_name}': {e}. Using deterministic semantic projection.")
                self._model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Computes L2-normalized dense embeddings for a batch of strings."""
        if self._model is not None:
            raw_embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(raw_embeddings, dtype=np.float32)

        # High-dimensional deterministic semantic projection fallback
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            embeddings[i] = self._deterministic_vector(text)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, formatted with BGE query instruction if supported."""
        if self._model is not None:
            formatted = f"Represent this sentence for searching relevant coaching passages: {query}"
            return self.embed_texts([formatted])[0]
        return self._deterministic_vector(query)

    def _deterministic_vector(self, text: str) -> np.ndarray:
        """Deterministic semantic vector generator with stopword filtering and n-gram projection."""
        import hashlib
        import re

        stopwords = {
            "the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "on", "at", "from",
            "by", "this", "that", "it", "as", "are", "was", "be", "or", "how", "what", "do",
            "does", "can", "should", "i", "you", "your", "my", "we", "our", "best", "way", "when",
            "presentation", "presentations", "speech", "speeches", "speaking", "speaker", "speakers",
            "coaching", "session", "module", "keynote", "drills", "practice", "remember"
        }

        # Tokenize alphanumeric words
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        content_words = [w for w in words if w not in stopwords and (len(w) > 2 or w in {"um", "uh"})]

        vec = np.zeros(self.dimension, dtype=np.float32)

        def project_token(token: str, weight: float):
            # Stable SHA-256 hash
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx1 = int.from_bytes(digest[:4], "big") % self.dimension
            idx2 = int.from_bytes(digest[4:8], "big") % self.dimension
            sign1 = 1.0 if digest[8] % 2 == 0 else -1.0
            sign2 = 1.0 if digest[9] % 2 == 0 else -1.0
            vec[idx1] += weight * sign1
            vec[idx2] += (weight * 0.5) * sign2

        # 1. Unigram projection
        for w in content_words:
            project_token(w, 1.0)

        # 2. Bigram projection for phrases like "filler words", "shoulder tilt", "words minute"
        for i in range(len(content_words) - 1):
            bigram = f"{content_words[i]}_{content_words[i+1]}"
            project_token(bigram, 1.8)

        # 3. Substring stems for inflected terms (e.g. "slouch", "postur", "fidget")
        for w in content_words:
            if len(w) >= 5:
                project_token(w[:5], 0.7)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec


class FallbackHNSWIndex:
    """Vector index fallback adhering to FAISS IndexHNSWFlat inner-product interface."""

    def __init__(self, dimension: int = 1024, m: int = 16):
        self.d = dimension
        self.m = m
        self.vectors: Optional[np.ndarray] = None
        self.efSearch = 32
        self.efConstruction = 64

    def add(self, x: np.ndarray) -> None:
        if self.vectors is None:
            self.vectors = np.array(x, dtype=np.float32, copy=True)
        else:
            self.vectors = np.vstack([self.vectors, x.astype(np.float32)])

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectors is None or len(self.vectors) == 0:
            return np.empty((len(queries), 0), dtype=np.float32), np.empty((len(queries), 0), dtype=np.int64)

        # Inner product on normalized vectors equals cosine similarity
        sims = np.dot(queries, self.vectors.T)
        k = min(k, len(self.vectors))
        indices = np.argsort(-sims, axis=1)[:, :k]
        distances = np.take_along_axis(sims, indices, axis=1)
        return distances.astype(np.float32), indices.astype(np.int64)

    @property
    def ntotal(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)


class FAISSHNSWIndexer:
    """Manages FAISS HNSW index (M=16) with efConstruction=64 and efSearch=32."""

    def __init__(
        self,
        dimension: int = 1024,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32
    ):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.embedder = BGEEmbedder(dimension=dimension)
        self.chunks: List[TranscriptChunk] = []
        self._index = self._create_index()

    def _create_index(self):
        """Creates FAISS IndexHNSWFlat with METRIC_INNER_PRODUCT."""
        if faiss is not None:
            index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index
        logger.info("Using FallbackHNSWIndex for vector indexing.")
        fallback = FallbackHNSWIndex(dimension=self.dimension, m=self.m)
        fallback.efConstruction = self.ef_construction
        fallback.efSearch = self.ef_search
        return fallback

    @property
    def total_indexed(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: List[TranscriptChunk]) -> None:
        """Embeds and indexes transcript chunks."""
        if not chunks:
            return

        texts = [
            f"Topic: {chunk.metadata.get('topic', '')}. {chunk.text}"
            for chunk in chunks
        ]
        embeddings = self.embedder.embed_texts(texts)

        # Normalize explicitly
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self._index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Indexed {len(chunks)} chunks. Total in index: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[TranscriptChunk, float]]:
        """Queries the HNSW graph and returns top-k matching chunks with similarity scores."""
        if len(self.chunks) == 0:
            return []

        q_vec = self.embedder.embed_query(query).reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        distances, indices = self._index.search(q_vec, top_k)
        results: List[Tuple[TranscriptChunk, float]] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results

    def save(self, directory: str) -> None:
        """Saves the index and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "chunks_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

        index_path = os.path.join(directory, "hnsw_index.bin")
        if faiss is not None and isinstance(self._index, faiss.Index):
            faiss.write_index(self._index, index_path)
        else:
            with open(index_path, "wb") as f:
                pickle.dump(self._index, f)
        logger.info(f"Saved RAG index with {len(self.chunks)} chunks to {directory}")

    def load(self, directory: str) -> bool:
        """Loads index and metadata from disk."""
        meta_path = os.path.join(directory, "chunks_metadata.json")
        index_path = os.path.join(directory, "hnsw_index.bin")

        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
            self.chunks = [
                TranscriptChunk(
                    chunk_id=c["chunk_id"],
                    transcript_id=c["transcript_id"],
                    text=c["text"],
                    token_count=c["token_count"],
                    start_token_idx=c["start_token_idx"],
                    end_token_idx=c["end_token_idx"],
                    metadata=c.get("metadata", {})
                )
                for c in raw_chunks
            ]

        if faiss is not None:
            try:
                self._index = faiss.read_index(index_path)
                return True
            except Exception:
                pass

        with open(index_path, "rb") as f:
            self._index = pickle.load(f)
        return True
