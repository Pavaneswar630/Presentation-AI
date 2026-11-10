"""Token-aware sliding window chunking engine for presentation transcripts."""

from dataclasses import dataclass, field
import re
from typing import List, Dict, Any, Optional


@dataclass(slots=True)
class TranscriptChunk:
    """A semantic chunk extracted from a speech coaching transcript."""
    chunk_id: str
    transcript_id: str
    text: str
    token_count: int
    start_token_idx: int
    end_token_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "transcript_id": self.transcript_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_token_idx": self.start_token_idx,
            "end_token_idx": self.end_token_idx,
            "metadata": self.metadata
        }


@dataclass(slots=True)
class TranscriptDocument:
    """Full transcript document with speaker and contextual metadata."""
    transcript_id: str
    title: str
    speaker: str
    topic: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TranscriptChunker:
    """Sliding-window chunker configured for 256-token chunk size and 32-token overlap."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap
        # Fast regex word-level tokenizer mimicking BPE token boundaries
        self._token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into word and punctuation tokens."""
        return self._token_pattern.findall(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Reconstructs text from tokens with proper spacing."""
        text = ""
        for token in tokens:
            if not text or re.match(r"[^\w\s]", token):
                text += token
            else:
                text += " " + token
        return text

    def chunk_document(self, document: TranscriptDocument) -> List[TranscriptChunk]:
        """Splits a document into overlapping 256-token chunks preserving metadata."""
        tokens = self.tokenize(document.content)
        total_tokens = len(tokens)
        chunks: List[TranscriptChunk] = []

        if total_tokens == 0:
            return chunks

        if total_tokens <= self.chunk_size:
            chunk = TranscriptChunk(
                chunk_id=f"{document.transcript_id}_chunk_0",
                transcript_id=document.transcript_id,
                text=document.content,
                token_count=total_tokens,
                start_token_idx=0,
                end_token_idx=total_tokens,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "speaker": document.speaker,
                    "topic": document.topic
                }
            )
            return [chunk]

        chunk_idx = 0
        for start_idx in range(0, total_tokens, self.step_size):
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.detokenize(chunk_tokens)

            chunk = TranscriptChunk(
                chunk_id=f"{document.transcript_id}_chunk_{chunk_idx}",
                transcript_id=document.transcript_id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                start_token_idx=start_idx,
                end_token_idx=end_idx,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "speaker": document.speaker,
                    "topic": document.topic,
                    "chunk_index": chunk_idx
                }
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_idx == total_tokens:
                break

        return chunks

    def chunk_corpus(self, corpus: List[TranscriptDocument]) -> List[TranscriptChunk]:
        """Batches chunking across a corpus of documents."""
        all_chunks: List[TranscriptChunk] = []
        for doc in corpus:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
