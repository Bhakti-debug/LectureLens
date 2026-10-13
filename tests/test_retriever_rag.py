"""
Tests for semantic lecture retriever and source attribution.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.schemas import LectureChunk
from rag.embeddings import get_embedding_service
from rag.retriever import LectureRetriever
from rag.vector_store import FAISSVectorStore


def test_retriever_citation_formatting_and_deduplication():
    """Verify citations include formatted timestamps, section titles, and deduplication."""
    emb_service = get_embedding_service()
    dim = emb_service.dimension

    chunks = [
        LectureChunk(
            chunk_id="c1",
            doc_id="d1",
            lecture_id="lec_1",
            section_number=1,
            section_title="Architecture Overview",
            start_time=65.0,
            end_time=120.0,
            text="The architecture combines V8 with C++ to provide asynchronous I/O.",
        ),
        LectureChunk(
            chunk_id="c2",
            doc_id="d1",
            lecture_id="lec_1",
            section_number=1,
            section_title="Architecture Overview",
            start_time=75.0,
            end_time=130.0,
            text="The architecture combines V8 with C++ to provide asynchronous I/O.",  # Duplicate text
        ),
        LectureChunk(
            chunk_id="c3",
            doc_id="d2",
            lecture_id="lec_1",
            section_number=2,
            section_title="Event Loop Details",
            start_time=130.0,
            end_time=200.0,
            text="The single-threaded event loop processes non-blocking network calls.",
        ),
    ]

    embeddings = emb_service.embed_texts([c.text for c in chunks])
    store = FAISSVectorStore("lec_1", dimension=dim)
    store.build_from_chunks(chunks, embeddings)

    retriever = LectureRetriever(vector_store=store, embedding_service=emb_service, top_k=3)

    results = retriever.retrieve("How does the architecture use V8 and C++?", top_k=2)

    # c1 and c2 duplicate text is deduplicated
    assert len(results) <= 2
    unique_ids = {r.chunk.chunk_id for r in results}
    assert not ("c1" in unique_ids and "c2" in unique_ids)

    citations = retriever.format_citations(results)
    assert len(citations) == len(results)
    first_cite = citations[0]
    assert first_cite.section_title == "Architecture Overview"
    assert first_cite.timestamp_formatted in ("01:05 - 02:00", "01:15 - 02:10")
    assert "architecture combines V8" in first_cite.snippet



def test_retriever_empty_query():
    """Verify retriever handles empty or whitespace queries safely."""
    emb_service = get_embedding_service()
    store = FAISSVectorStore("lec_empty", dimension=emb_service.dimension)
    retriever = LectureRetriever(vector_store=store, embedding_service=emb_service)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
