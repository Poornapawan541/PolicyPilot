import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embedder import EmbeddingModel


def main():
    model = EmbeddingModel()

    # Test single text embedding
    single_text = "Employees are entitled to annual leave."
    embedding = model.embed_text(single_text)

    # Validations for single text embedding
    if not isinstance(embedding, list):
        print("Error: Single embedding is not a list.")
        return

    if len(embedding) == 0:
        print("Error: Single embedding is empty.")
        return

    print("Single embedding generated successfully.")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch text embedding
    batch_texts = [
        "Employees are entitled to annual leave.",
        "Employees must follow the company password policy.",
    ]
    batch_embeddings = model.embed_texts(batch_texts)

    # Validations for batch text embeddings
    if len(batch_embeddings) != 2:
        print(f"Error: Expected 2 embeddings, got {len(batch_embeddings)}.")
        return

    dim_1 = len(batch_embeddings[0])
    dim_2 = len(batch_embeddings[1])

    if dim_1 != dim_2:
        print(f"Error: Dimension mismatch in batch embeddings: {dim_1} vs {dim_2}.")
        return

    print("\nBatch embedding test passed.")
    print(f"Number of embeddings: {len(batch_embeddings)}")
    print(f"Embedding dimension: {dim_1}")


if __name__ == "__main__":
    main()
