"""
Tests for local embedding generation and FAISS vector store isolation.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.schemas import LectureChunk
from rag.embeddings import get_embedding_service
from rag.vector_store import FAISSVectorStore


def test_embedding_dimensions_and_normalization():
    """Verify embeddings produce 384-dimensional unit vectors (L2 normalized)."""
    emb_service = get_embedding_service()
    assert emb_service.dimension == 384

    texts = ["Node.js is an open-source runtime.", "Chrome uses the V8 engine."]
    embeddings = emb_service.embed_texts(texts)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify unit length (L2 norm is approximately 1.0 for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_vector_store_build_search_and_persistence():
    """Verify index creation, top-k retrieval, saving, and loading."""
    emb_service = get_embedding_service()
    dim = emb_service.dimension

    chunks = [
        LectureChunk(
            chunk_id="chk_01",
            doc_id="doc_01",
            lecture_id="lec_alpha",
            section_number=1,
            section_title="Alpha Section",
            start_time=0.0,
            end_time=30.0,
            text="Understanding web browser engines like Chrome V8.",
        ),
        LectureChunk(
            chunk_id="chk_02",
            doc_id="doc_02",
            lecture_id="lec_alpha",
            section_number=2,
            section_title="Beta Section",
            start_time=30.0,
            end_time=60.0,
            text="Baking bread and sourdough recipes in the kitchen.",
        ),
    ]

    embeddings = emb_service.embed_texts([c.text for c in chunks])

    store = FAISSVectorStore(
        lecture_id="lec_alpha",
        dimension=dim,
        model_name=emb_service.model_name,
    )
    store.build_from_chunks(chunks, embeddings)

    # Search for V8
    q_vec = emb_service.embed_query("What engine does Chrome run?")
    results = store.search(q_vec, top_k=2)

    assert len(results) == 2
    # The browser engine chunk must rank higher than sourdough bread
    assert results[0].chunk.chunk_id == "chk_01"
    assert results[0].score > results[1].score

    # Save and Load
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "alpha_index"
        store.save(save_path)

        loaded_store = FAISSVectorStore.load(
            save_path,
            expected_lecture_id="lec_alpha",
            expected_dimension=dim,
        )

        assert loaded_store is not None
        assert len(loaded_store.chunks) == 2
        assert loaded_store.index.ntotal == 2

        # Re-search on loaded store
        loaded_res = loaded_store.search(q_vec, top_k=1)
        assert loaded_res[0].chunk.chunk_id == "chk_01"


def test_vector_store_lecture_isolation():
    """Verify vector store rejects loading index belonging to another lecture."""
    emb_service = get_embedding_service()
    dim = emb_service.dimension

    chunks = [
        LectureChunk(
            chunk_id="chk_01",
            doc_id="doc_01",
            lecture_id="lec_first",
            section_number=1,
            section_title="First",
            start_time=0.0,
            end_time=10.0,
            text="First lecture content.",
        )
    ]
    embeddings = emb_service.embed_texts([c.text for c in chunks])

    store = FAISSVectorStore(
        lecture_id="lec_first",
        dimension=dim,
        model_name=emb_service.model_name,
    )
    store.build_from_chunks(chunks, embeddings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "first_index"
        store.save(save_path)

        # Attempt to load with a different expected lecture_id
        mismatched_store = FAISSVectorStore.load(
            save_path,
            expected_lecture_id="lec_second",
            expected_dimension=dim,
        )

        assert mismatched_store is None
