"""Unit and integration tests for the Offline RAG Pipeline."""

import pytest
from rag_engine.chunker import TranscriptDocument, TranscriptChunker
from rag_engine.indexer import FAISSHNSWIndexer
from rag_engine.retriever import CoachingRetriever
from rag_engine.corpus_generator import TranscriptCorpusGenerator
from rag_engine.evaluation import RAGEvaluationHarness


def test_transcript_chunker_sizes_and_overlap():
    """Verify that chunker strictly respects 256-token size and 32-token overlap."""
    chunker = TranscriptChunker(chunk_size=256, chunk_overlap=32)
    
    # Generate content with exactly 500 words
    words = [f"word_{i}" for i in range(500)]
    doc = TranscriptDocument(
        transcript_id="doc_test_1",
        title="Pacing Workshop",
        speaker="Speaker A",
        topic="Cadence",
        content=" ".join(words)
    )
    
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    
    # First chunk should have at most 256 tokens
    assert chunks[0].token_count <= 256
    assert chunks[0].start_token_idx == 0
    assert chunks[0].end_token_idx <= 256
    
    # Second chunk should start at 256 - 32 = 224
    assert chunks[1].start_token_idx == 224
    assert chunks[1].metadata["title"] == "Pacing Workshop"


def test_faiss_hnsw_indexer_indexing_and_search():
    """Verify that FAISS HNSW M=16 indexer embeds, indexes, and queries vectors."""
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    chunker = TranscriptChunker(chunk_size=256, chunk_overlap=32)
    
    docs = [
        TranscriptDocument(
            transcript_id="posture_doc",
            title="Spinal Alignment",
            speaker="Coach",
            topic="Posture",
            content="Maintain an upright spine and level shoulders to project authority."
        ),
        TranscriptDocument(
            transcript_id="speed_doc",
            title="Speech Speed",
            speaker="Coach",
            topic="Cadence",
            content="Regulate speech to 140 words per minute and use deliberate pausing."
        )
    ]
    
    chunks = chunker.chunk_corpus(docs)
    indexer.add_chunks(chunks)
    assert indexer.total_indexed == 2
    
    # Search for posture
    results = indexer.search(query="How to keep my spine straight?", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert "spine" in top_chunk.text.lower() or "shoulders" in top_chunk.text.lower()


def test_coaching_retriever_defect_mapping():
    """Verify CoachingRetriever maps presentation defects to coaching guidance."""
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    chunker = TranscriptChunker(chunk_size=256, chunk_overlap=32)
    
    doc = TranscriptDocument(
        transcript_id="doc_fillers",
        title="Eliminating Fillers",
        speaker="Coach",
        topic="Speech Disfluencies",
        content="Replace vocalized fillers like um and uh with silent two-second pauses."
    )
    indexer.add_chunks(chunker.chunk_corpus([doc]))
    
    retriever = CoachingRetriever(indexer)
    advice_list = retriever.retrieve_for_defect("FILLER_WORD_CLUSTER", top_k=1)
    
    assert len(advice_list) == 1
    assert "um" in advice_list[0].recommendation.lower() or "filler" in advice_list[0].recommendation.lower()


def test_rag_evaluation_harness_hit_rate_sla():
    """Verify that the evaluation harness over 1,000 transcripts achieves Hit Rate@3 >= 0.82."""
    corpus, test_pairs = TranscriptCorpusGenerator.generate_corpus(count=1000)
    chunker = TranscriptChunker(chunk_size=256, chunk_overlap=32)
    indexer = FAISSHNSWIndexer(dimension=1024, m=16, ef_construction=64, ef_search=32)
    
    harness = RAGEvaluationHarness(indexer=indexer, chunker=chunker, target_hit_rate_at_3=0.82)
    
    total_chunks = harness.build_and_index_corpus(corpus)
    assert total_chunks >= 1000
    
    # Evaluate over test queries
    report = harness.evaluate(test_pairs, top_k=3)
    
    assert report["hit_rate_at_3"] >= 0.82
    assert report["sla_achieved"] is True
    assert report["mrr_at_3"] > 0.70
    assert report["average_latency_ms"] < 25.0
