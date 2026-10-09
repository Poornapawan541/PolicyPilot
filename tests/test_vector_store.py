import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_store import VectorStore


def main():
    store = VectorStore(dimension=384)

    # Create two simple test vectors.
    vector_1 = [1.0] + [0.0] * 383
    vector_2 = [0.0, 1.0] + [0.0] * 382

    store.add_embeddings([vector_1, vector_2])

    print(f"Vectors stored: {store.count()}")

    if store.count() != 2:
        print("Error: Expected 2 vectors.")
        return

    # Search using a query close to vector_1.
    query_vector = [1.0] + [0.0] * 383

    scores, indices = store.search(
        query_vector,
        top_k=2,
    )

    print(f"Search indices: {indices.tolist()}")
    print(f"Search scores: {scores.tolist()}")

    if indices[0] != 0:
        print("Error: Expected vector 1 to be the closest result.")
        return

    print("\nVector store test passed.")


if __name__ == "__main__":
    main()