"""
LectureLens - Embedding Service
Manages local Sentence Transformer embedding generation, vector normalization,
model caching, and dimensionality tracking.
"""

from typing import List, Optional
import numpy as np

from utils.config import get_config
from utils.logging_config import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    """Singleton service for generating normalized text embeddings."""

    _instance: Optional["EmbeddingService"] = None
    _model = None
    _model_name: str = ""
    _dimension: int = 384

    def __new__(cls, model_name: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cfg = get_config()
            target_model = model_name or cfg.embedding_model
            cls._instance._init_model(target_model)
        return cls._instance

    def _init_model(self, model_name: str) -> None:
        """Initialize and load the sentence transformer model once."""
        self._model_name = model_name
        logger.info(f"Loading embedding model: {model_name}...")
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            test_vec = self._model.encode(["dimension_probe"], normalize_embeddings=True)
            self._dimension = int(test_vec.shape[1])
            logger.info(f"Embedding model loaded successfully. Dimension: {self._dimension}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer ({model_name}): {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text strings with L2 normalization for cosine similarity.

        Returns:
            np.ndarray of shape (len(texts), dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        # Sanitize texts (replace empty or whitespace-only with placeholder)
        cleaned = [t.strip() if t and t.strip() else "[EMPTY]" for t in texts]

        embeddings = self._model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            np.ndarray of shape (1, dimension) and dtype float32.
        """
        cleaned = query.strip() if query and query.strip() else "[EMPTY]"
        emb = self._model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)


def get_embedding_service() -> EmbeddingService:
    """Return the global embedding service instance."""
    return EmbeddingService()
