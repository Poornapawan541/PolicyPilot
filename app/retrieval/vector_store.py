import faiss
import numpy as np


class VectorStore:
    """
    Local FAISS vector store for semantic search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

        # Inner product works as cosine similarity because
        # our embeddings are normalized.
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: list[list[float]]) -> None:
        """
        Add embedding vectors to the FAISS index.
        """
        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D list.")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        self.index.add(vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the most similar vectors.

        Returns:
            scores and vector positions.
        """
        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"got {query_vector.shape[1]}."
            )

        scores, indices = self.index.search(query_vector, top_k)

        return scores[0], indices[0]

    def count(self) -> int:
        """
        Return the number of vectors stored.
        """
        return self.index.ntotal