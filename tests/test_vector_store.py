import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_store import VectorStore


def test_vector_store_empty_index():
    store = VectorStore(dimension=4)
    assert store.count() == 0

    # Search on empty index
    scores, indices = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(scores) == 3
    assert len(indices) == 3
    assert all(idx == -1 for idx in indices)
    assert all(np.isneginf(s) for s in scores)


def test_vector_store_exact_similarity_and_ranking():
    store = VectorStore(dimension=4)
    # Unit vectors
    v0 = [1.0, 0.0, 0.0, 0.0]
    v1 = [0.0, 1.0, 0.0, 0.0]
    v2 = [0.6, 0.8, 0.0, 0.0]  # dot with v0 is 0.6, dot with v1 is 0.8

    store.add_embeddings([v0, v1, v2])
    assert store.count() == 3

    # Query with v0: v0 dot v0 = 1.0, v2 dot v0 = 0.6, v1 dot v0 = 0.0
    scores, indices = store.search(v0, top_k=3)
    assert list(indices) == [0, 2, 1]
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    assert np.isclose(scores[1], 0.6, atol=1e-5)
    assert np.isclose(scores[2], 0.0, atol=1e-5)


def test_vector_store_top_k_larger_than_count():
    store = VectorStore(dimension=3)
    store.add_embeddings([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert store.count() == 2

    scores, indices = store.search([1.0, 0.0, 0.0], top_k=4)
    assert len(indices) == 4
    assert list(indices[:2]) == [0, 1]
    assert list(indices[2:]) == [-1, -1]
    assert np.isneginf(scores[2]) and np.isneginf(scores[3])


def test_vector_store_tie_breaking_deterministic():
    store = VectorStore(dimension=2)
    # Two identical vectors
    v = [1.0, 0.0]
    store.add_embeddings([v, v])

    scores, indices = store.search(v, top_k=2)
    # Lower original index wins on tie
    assert list(indices) == [0, 1]
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 1.0)


def test_vector_store_dimension_validation():
    store = VectorStore(dimension=3)
    with pytest.raises(ValueError, match="Expected embedding dimension 3"):
        store.add_embeddings([[1.0, 0.0]])

    store.add_embeddings([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Expected query dimension 3"):
        store.search([1.0, 0.0])


def main():
    test_vector_store_empty_index()
    test_vector_store_exact_similarity_and_ranking()
    test_vector_store_top_k_larger_than_count()
    test_vector_store_tie_breaking_deterministic()
    test_vector_store_dimension_validation()
    print("All VectorStore tests passed successfully.")


if __name__ == "__main__":
    main()