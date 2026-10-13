"""
LectureLens - Complete RAG Pipeline Orchestrator
Coordinates document building, vector indexing, semantic retrieval,
relevance verification, and grounded answer synthesis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import RAGResponse
from rag.answer_generator import AnswerGenerator
from rag.chunker import get_default_chunker
from rag.document_builder import build_lecture_documents
from rag.embeddings import get_embedding_service
from rag.relevance import RelevanceGuard
from rag.retriever import LectureRetriever
from rag.vector_store import FAISSVectorStore
from utils.config import get_config
from utils.logging_config import get_logger

logger = get_logger("rag_pipeline")


class RAGPipeline:
    """End-to-end Lecture RAG pipeline."""

    def __init__(self, lecture_id: str, lecture_title: str = "Active Lecture"):
        self.lecture_id = lecture_id
        self.lecture_title = lecture_title
        self.config = get_config()
        self.embedding_service = get_embedding_service()
        self.chunker = get_default_chunker()
        self.relevance_guard = RelevanceGuard()
        self.answer_generator = AnswerGenerator()

        self.vector_store: Optional[FAISSVectorStore] = None
        self.retriever: Optional[LectureRetriever] = None
        self.lecture_overview: str = ""
        self.key_topics: List[str] = []

        # Attempt to load existing index for this lecture
        self._load_or_init_store()

    def _get_index_dir(self) -> Path:
        return self.config.indexes_dir / self.lecture_id

    def _load_or_init_store(self) -> None:
        index_dir = self._get_index_dir()
        if index_dir.exists():
            store = FAISSVectorStore.load(
                index_dir,
                expected_lecture_id=self.lecture_id,
                expected_dimension=self.embedding_service.dimension,
            )
            if store:
                self.vector_store = store
                self.retriever = LectureRetriever(
                    vector_store=store,
                    embedding_service=self.embedding_service,
                    top_k=self.config.retrieval_top_k,
                )
                logger.info(f"Loaded existing index for lecture: {self.lecture_id}")

    def index_lecture(
        self,
        analysis_data: Dict[str, Any],
        notes_data: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Build and save the FAISS vector index from lecture analysis data.
        Returns the number of indexed chunks.
        """
        index_dir = self._get_index_dir()
        if not force_reindex and self.vector_store is not None and self.vector_store.chunks:
            logger.info(f"Index already loaded for {self.lecture_id}. Skipping re-indexing.")
            return len(self.vector_store.chunks)

        self.lecture_overview = analysis_data.get("video_summary", "")

        # Extract topics
        topics = set()
        for sec in analysis_data.get("sections", []):
            for c in sec.get("key_concepts", []):
                topics.add(str(c))
        self.key_topics = list(topics)

        # 1. Build documents
        documents = build_lecture_documents(
            analysis_data=analysis_data,
            lecture_id=self.lecture_id,
            notes_data=notes_data,
        )

        # 2. Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from lecture documents.")
            return 0

        # 3. Embed chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # 4. Build and save FAISS store
        store = FAISSVectorStore(
            lecture_id=self.lecture_id,
            dimension=self.embedding_service.dimension,
            model_name=self.embedding_service.model_name,
        )
        store.build_from_chunks(chunks=chunks, embeddings=embeddings)
        store.save(index_dir)

        self.vector_store = store
        self.retriever = LectureRetriever(
            vector_store=store,
            embedding_service=self.embedding_service,
            top_k=self.config.retrieval_top_k,
        )

        return len(chunks)

    def ask(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """
        Process a user question through retrieval, relevance guard, and answer generation.
        """
        if self.retriever is None:
            return RAGResponse(
                query=query,
                answer="No lecture index is currently loaded. Please load or analyze a lecture first.",
                is_relevant=False,
                is_sufficient=False,
                refusal_reason="Index missing",
            )

        # 1. Semantic Retrieval
        retrieved_results = self.retriever.retrieve(query)
        citations = self.retriever.format_citations(retrieved_results)

        # 2. Relevance Guard Decision
        relevance_decision = self.relevance_guard.evaluate_query(
            question=query,
            lecture_title=self.lecture_title,
            lecture_overview=self.lecture_overview,
            key_topics=self.key_topics,
            retrieved_results=retrieved_results,
        )

        if not relevance_decision["is_relevant"]:
            return RAGResponse(
                query=query,
                answer=relevance_decision["refusal_message"] or "This question is not relevant to this lecture.",
                is_relevant=False,
                is_sufficient=False,
                sources=[],
                relevance_reason=relevance_decision["reason"],
                refusal_reason=relevance_decision["reason"],
            )

        is_insufficient = relevance_decision["category"] == "insufficient_evidence"

        # 3. Grounded Answer Synthesis
        gen_result = self.answer_generator.generate_answer(
            question=query,
            lecture_title=self.lecture_title,
            retrieved_results=retrieved_results,
            chat_history=chat_history,
            is_insufficient_evidence=is_insufficient,
        )

        return RAGResponse(
            query=query,
            answer=gen_result["answer"],
            is_relevant=True,
            is_sufficient=gen_result["is_sufficient"],
            sources=citations,
            relevance_reason=relevance_decision["reason"],
            refusal_reason="",
        )
