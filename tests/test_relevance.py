"""
Tests for lecture relevance boundary and unrelated question refusal.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.schemas import LectureChunk, RetrievalResult
from rag.relevance import RelevanceGuard


def test_relevance_empty_question():
    """Verify empty questions are rejected immediately."""
    guard = RelevanceGuard()
    res = guard.evaluate_query(
        question="",
        lecture_title="Node.js Intro",
        lecture_overview="Overview of Node.js",
        key_topics=["Node.js", "V8"],
        retrieved_results=[],
    )
    assert res["is_relevant"] is False
    assert res["category"] == "out_of_scope"


def test_relevance_extreme_out_of_scope_fast_path():
    """Verify unrelated questions with near-zero similarity are rejected by the boundary."""
    guard = RelevanceGuard()

    # Create dummy retrieval results with very low scores (< 0.10)
    dummy_chunk = LectureChunk(
        chunk_id="c1",
        doc_id="d1",
        lecture_id="lec_1",
        section_number=1,
        section_title="V8 Engine",
        start_time=0.0,
        end_time=30.0,
        text="Node.js and the V8 JavaScript engine.",
    )
    low_res = [RetrievalResult(chunk=dummy_chunk, score=0.05)]

    res = guard.evaluate_query(
        question="What is the score in the cricket world cup final?",
        lecture_title="Node.js Intro",
        lecture_overview="Node.js runtime explanation",
        key_topics=["Node.js", "V8", "C++"],
        retrieved_results=low_res,
    )
    assert res["is_relevant"] is False
    assert res["category"] == "out_of_scope"
    assert "outside the scope" in res["refusal_message"]


def test_relevance_in_scope_with_good_retrieval_score():
    """Verify relevant query matching key concepts is accepted."""
    guard = RelevanceGuard(similarity_threshold=0.35)

    dummy_chunk = LectureChunk(
        chunk_id="c1",
        doc_id="d1",
        lecture_id="lec_1",
        section_number=1,
        section_title="V8 Engine",
        start_time=0.0,
        end_time=30.0,
        text="V8 engine executes JavaScript inside Google Chrome.",
    )
    good_res = [RetrievalResult(chunk=dummy_chunk, score=0.78)]

    # Mock or test evaluation
    res = guard.evaluate_query(
        question="What is Chrome's V8 engine?",
        lecture_title="Node.js Intro",
        lecture_overview="Node.js and V8",
        key_topics=["Node.js", "V8", "Chrome"],
        retrieved_results=good_res,
    )
    assert res["is_relevant"] is True
    assert res["refusal_message"] is None
