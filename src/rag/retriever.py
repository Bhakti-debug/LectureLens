"""
LectureLens - Semantic Lecture Retriever
Executes similarity search over the active lecture's FAISS index,
deduplicates redundant passages, and formats source citations with timestamps.
"""

from typing import List, Optional
from models.schemas import RetrievalResult, SourceCitation
from rag.embeddings import EmbeddingService, get_embedding_service
from rag.vector_store import FAISSVectorStore
from utils.config import get_config
from utils.logging_config import get_logger
from utils.validation import format_timestamp

logger = get_logger("retriever")


class LectureRetriever:
    """Semantic retriever operating strictly within a lecture's FAISS store."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: Optional[int] = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        cfg = get_config()
        self.top_k = top_k or cfg.retrieval_top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve the top-k passages for a student question.
        Deduplicates near-identical chunks.
        """
        k = top_k or self.top_k
        if not query or not query.strip():
            return []

        q_vec = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search(q_vec, top_k=k * 2)

        # Deduplicate passages with substantial text overlap
        unique_results: List[RetrievalResult] = []
        seen_texts = set()

        for res in raw_results:
            normalized_snippet = " ".join(res.chunk.text.split()[:20]).lower()
            if normalized_snippet in seen_texts:
                continue
            seen_texts.add(normalized_snippet)
            unique_results.append(res)
            if len(unique_results) >= k:
                break

        logger.debug(
            f"Query: '{query[:40]}...' retrieved {len(unique_results)} passages (scores: {[round(r.score, 3) for r in unique_results]})."
        )
        return unique_results

    def format_citations(self, results: List[RetrievalResult]) -> List[SourceCitation]:
        """Format retrieved passages into structured citation objects."""
        citations: List[SourceCitation] = []
        for res in results:
            chunk = res.chunk
            ts_str = f"{format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}"

            # Take a concise excerpt for the snippet
            lines = [l for l in chunk.text.split("\n") if l.strip()]
            snippet = lines[0] if lines else chunk.text[:120]
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."

            citations.append(
                SourceCitation(
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    timestamp_formatted=ts_str,
                    snippet=snippet,
                    score=round(res.score, 4),
                )
            )
        return citations
