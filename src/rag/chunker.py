"""
LectureLens - Topic-Aware Deterministic Text Chunker
Splits lecture documents into contextual, retrieval-sized passages while preserving
sentence boundaries, section metadata, and timestamps.
"""

import re
from typing import List
from models.schemas import LectureChunk, LectureDocument
from utils.config import get_config


class LectureChunker:
    """Deterministic chunker preserving topic and sentence coherence."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        """
        chunk_size: Target size in approximate words (~1.3 tokens/word).
        chunk_overlap: Overlap in approximate words.
        """
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences or bullet points without breaking definitions."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        units: List[str] = []

        for line in lines:
            if line.startswith("-") or line.startswith("*") or len(line) < 80:
                units.append(line)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", line)
                units.extend([s.strip() for s in sentences if s.strip()])

        return units

    def chunk_document(self, doc: LectureDocument) -> List[LectureChunk]:
        """Chunk a single lecture document into LectureChunk items."""
        if not doc.text or not doc.text.strip():
            return []

        units = self._split_into_sentences(doc.text)
        if not units:
            return []

        chunks: List[LectureChunk] = []
        current_words: List[str] = []
        current_text_units: List[str] = []
        chunk_idx = 0

        for unit in units:
            unit_words = unit.split()
            potential_length = len(current_words) + len(unit_words)

            if potential_length > self.chunk_size and current_words:
                chunk_text = "\n".join(current_text_units).strip()
                chunks.append(
                    LectureChunk(
                        chunk_id=f"{doc.doc_id}_chk_{chunk_idx:02d}",
                        doc_id=doc.doc_id,
                        lecture_id=doc.lecture_id,
                        section_number=doc.section_number,
                        section_title=doc.section_title,
                        start_time=doc.start_time,
                        end_time=doc.end_time,
                        text=chunk_text,
                        source_type=doc.source_type,
                        chunk_index=chunk_idx,
                        token_count=len(current_words),
                        metadata=dict(doc.metadata),
                    )
                )
                chunk_idx += 1

                # Retain overlap words from the end
                if self.chunk_overlap > 0:
                    overlap_units: List[str] = []
                    overlap_word_count = 0
                    for u in reversed(current_text_units):
                        w_count = len(u.split())
                        if overlap_word_count + w_count <= self.chunk_overlap:
                            overlap_units.insert(0, u)
                            overlap_word_count += w_count
                        else:
                            break
                    current_text_units = overlap_units + [unit]
                    current_words = " ".join(current_text_units).split()
                else:
                    current_text_units = [unit]
                    current_words = list(unit_words)
            else:
                current_text_units.append(unit)
                current_words.extend(unit_words)

        if current_text_units:
            chunk_text = "\n".join(current_text_units).strip()
            chunks.append(
                LectureChunk(
                    chunk_id=f"{doc.doc_id}_chk_{chunk_idx:02d}",
                    doc_id=doc.doc_id,
                    lecture_id=doc.lecture_id,
                    section_number=doc.section_number,
                    section_title=doc.section_title,
                    start_time=doc.start_time,
                    end_time=doc.end_time,
                    text=chunk_text,
                    source_type=doc.source_type,
                    chunk_index=chunk_idx,
                    token_count=len(current_words),
                    metadata=dict(doc.metadata),
                )
            )

        return chunks

    def chunk_documents(self, documents: List[LectureDocument]) -> List[LectureChunk]:
        """Process multiple documents and return all chunks."""
        all_chunks: List[LectureChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def get_default_chunker() -> LectureChunker:
    """Return a chunker configured from global AppConfig."""
    cfg = get_config()
    return LectureChunker(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
