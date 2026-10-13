"""
LectureLens - FAISS Vector Store
Manages lecture-isolated FAISS indexes, persistent chunk metadata,
index validation, and similarity search.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from models.schemas import LectureChunk, RetrievalResult
from utils.logging_config import get_logger

logger = get_logger("vector_store")


class FAISSVectorStore:
    """Lecture-isolated FAISS index with synchronized chunk metadata."""

    def __init__(
        self,
        lecture_id: str,
        dimension: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.lecture_id = lecture_id
        self.dimension = dimension
        self.model_name = model_name
        self.index: Optional[faiss.Index] = None
        self.chunks: List[LectureChunk] = []
        self._content_hash: str = ""

    def build_from_chunks(
        self,
        chunks: List[LectureChunk],
        embeddings: np.ndarray,
        content_hash: str = "",
    ) -> None:
        """
        Build a FAISS Inner Product index from normalized embeddings and chunks.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            logger.warning(f"No chunks to index for lecture {self.lecture_id}.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self._content_hash = content_hash
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embeddings count ({embeddings.shape[0]})"
            )

        # Create IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        # Ensure float32 C-contiguous
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        index.add(vecs)

        self.index = index
        self.chunks = list(chunks)
        self._content_hash = content_hash
        logger.info(
            f"Successfully built FAISS index for lecture '{self.lecture_id}' with {len(chunks)} chunks."
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
    ) -> List[RetrievalResult]:
        """
        Search the index for the most similar chunks.

        Returns:
            List[RetrievalResult] ordered by descending similarity score.
        """
        if self.index is None or self.index.ntotal == 0 or not self.chunks:
            return []

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return results

    def save(self, directory: Path) -> Path:
        """
        Save the FAISS index and chunk metadata atomically to disk.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.json"

        if self.index is not None:
            faiss.write_index(self.index, str(index_file))

        metadata = {
            "lecture_id": self.lecture_id,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "content_hash": self._content_hash,
            "chunk_count": len(self.chunks),
            "chunks": [c.model_dump() for c in self.chunks],
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index and metadata to {directory}")
        return directory

    @classmethod
    def load(
        cls,
        directory: Path,
        expected_lecture_id: Optional[str] = None,
        expected_dimension: Optional[int] = None,
    ) -> Optional["FAISSVectorStore"]:
        """
        Load an existing FAISS index and verify compatibility.
        """
        directory = Path(directory)
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            logger.warning(f"Index or metadata missing in {directory}")
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            lecture_id = meta.get("lecture_id", "")
            if expected_lecture_id and lecture_id != expected_lecture_id:
                logger.error(
                    f"Lecture ID mismatch in index: expected {expected_lecture_id}, found {lecture_id}"
                )
                return None

            dimension = int(meta.get("dimension", 384))
            if expected_dimension and dimension != expected_dimension:
                logger.error(
                    f"Dimension mismatch in index: expected {expected_dimension}, found {dimension}"
                )
                return None

            model_name = meta.get("model_name", "all-MiniLM-L6-v2")
            content_hash = meta.get("content_hash", "")

            # Load raw chunks
            raw_chunks = meta.get("chunks", [])
            chunks = [LectureChunk(**c) for c in raw_chunks]

            # Load FAISS index
            index = faiss.read_index(str(index_file))

            if index.ntotal != len(chunks):
                logger.error(
                    f"Index corruption: FAISS count ({index.ntotal}) != chunks count ({len(chunks)})"
                )
                return None

            store = cls(
                lecture_id=lecture_id,
                dimension=dimension,
                model_name=model_name,
            )
            store.index = index
            store.chunks = chunks
            store._content_hash = content_hash

            logger.info(
                f"Successfully loaded index for lecture '{lecture_id}' ({len(chunks)} chunks)."
            )
            return store

        except Exception as e:
            logger.error(f"Failed to load vector store from {directory}: {e}")
            return None
