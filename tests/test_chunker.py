"""
Tests for document building and deterministic topic-aware chunking.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rag.chunker import LectureChunker
from rag.document_builder import build_lecture_documents
from models.schemas import LectureDocument


def test_build_lecture_documents():
    """Verify document building preserves metadata, spoken content, and visuals."""
    sample_analysis = {
        "video_summary": "Introduction to programming concepts.",
        "sections": [
            {
                "title": "Introduction to Variables",
                "start_time": 10.0,
                "end_time": 60.0,
                "spoken_content": [
                    "A variable stores a value in memory.",
                    "Variables have types such as integer and string.",
                ],
                "visual_content": [
                    {
                        "type": "diagram",
                        "timestamp": 25.0,
                        "description": "Memory allocation chart",
                        "visible_text": "x = 42",
                    }
                ],
                "audio_visual_relationship": [
                    "Instructor highlights the memory box for variable x."
                ],
                "key_concepts": ["Variables", "Memory", "Data Types"],
            }
        ],
    }

    docs = build_lecture_documents(sample_analysis, lecture_id="lec_test")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.lecture_id == "lec_test"
    assert doc.section_number == 1
    assert doc.section_title == "Introduction to Variables"
    assert doc.start_time == 10.0
    assert doc.end_time == 60.0
    assert "Spoken Lecture Content:" in doc.text
    assert "Memory allocation chart" in doc.text
    assert 'Text on screen: "x = 42"' in doc.text


def test_chunker_deterministic_and_metadata():
    """Verify chunker outputs deterministic chunks and maintains metadata."""
    chunker = LectureChunker(chunk_size=30, chunk_overlap=10)

    doc = LectureDocument(
        doc_id="doc_1",
        lecture_id="lec_1",
        section_number=1,
        section_title="Sample Title",
        start_time=15.0,
        end_time=80.0,
        text=(
            "Sentence one introduces the core topic. "
            "Sentence two explains the mechanism of execution. "
            "Sentence three provides a concrete coding example. "
            "Sentence four discusses the performance implications. "
            "Sentence five summarizes the final takeaway."
        ),
    )

    chunks1 = chunker.chunk_document(doc)
    chunks2 = chunker.chunk_document(doc)

    assert len(chunks1) > 0
    # Determinism
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.text == c2.text
        assert c1.chunk_id == c2.chunk_id
        assert c1.section_number == 1
        assert c1.section_title == "Sample Title"
        assert c1.start_time == 15.0
        assert c1.end_time == 80.0


def test_chunker_empty_and_unicode():
    """Verify chunker handles empty text and unicode characters safely."""
    chunker = LectureChunker(chunk_size=100, chunk_overlap=20)

    empty_doc = LectureDocument(
        doc_id="doc_empty",
        lecture_id="lec_1",
        section_number=1,
        section_title="Empty",
        start_time=0.0,
        end_time=10.0,
        text="",
    )
    assert chunker.chunk_document(empty_doc) == []

    unicode_doc = LectureDocument(
        doc_id="doc_unicode",
        lecture_id="lec_1",
        section_number=2,
        section_title="Unicode",
        start_time=10.0,
        end_time=30.0,
        text="JavaScript supports emojis 🚀 and international strings like 'こんにちは世界'.",
    )
    chunks = chunker.chunk_document(unicode_doc)
    assert len(chunks) == 1
    assert "🚀" in chunks[0].text
    assert "こんにちは世界" in chunks[0].text
