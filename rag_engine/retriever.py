"""Coaching advice retriever for real-time and post-session diagnostic prompts."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .indexer import FAISSHNSWIndexer
from .chunker import TranscriptChunk


@dataclass(slots=True)
class CoachingAdvice:
    """Formatted recommendation for the speaker."""
    source_chunk_id: str
    topic: str
    similarity_score: float
    recommendation: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_chunk_id": self.source_chunk_id,
            "topic": self.topic,
            "similarity_score": round(self.similarity_score, 3),
            "recommendation": self.recommendation,
            "metadata": self.metadata
        }


class CoachingRetriever:
    """Queries the indexed coaching corpus based on presentation defect profiles."""

    def __init__(self, indexer: FAISSHNSWIndexer):
        self.indexer = indexer

    def retrieve_for_defect(
        self,
        defect_type: str,
        context_details: Optional[str] = None,
        top_k: int = 3
    ) -> List[CoachingAdvice]:
        """Translates a defect type into a semantic query and fetches coaching guidance."""
        query_map = {
            "SLUMPED_POSTURE": "Techniques for maintaining an upright, confident spinal posture and aligned shoulders during keynotes.",
            "SHOULDER_ASYMMETRY": "Correcting uneven shoulder tilt, body stance balance, and weight distribution while presenting.",
            "EXCESSIVE_FIDGETING": "Calming nervous hand gestures, anchoring hands at the waist, and deliberate purposeful gestures.",
            "RAPID_PACING": "Slowing down speech cadence, using deliberate pauses, and managing breath pacing.",
            "FILLER_WORD_CLUSTER": "Eliminating filler words like 'um', 'uh', and 'like' using silent pausing techniques.",
            "MONOTONE_DELIVERY": "Vocal pitch modulation, emphasizing key verbs and numbers, and dynamic volume variation."
        }

        query = query_map.get(defect_type, f"Coaching advice for improving {defect_type.lower().replace('_', ' ')}")
        if context_details:
            query = f"{query} Context: {context_details}"

        results = self.indexer.search(query=query, top_k=top_k)
        recommendations: List[CoachingAdvice] = []

        for chunk, score in results:
            topic = chunk.metadata.get("topic", "General Presentation Delivery")
            recommendations.append(
                CoachingAdvice(
                    source_chunk_id=chunk.chunk_id,
                    topic=topic,
                    similarity_score=score,
                    recommendation=chunk.text,
                    metadata=chunk.metadata
                )
            )

        return recommendations
