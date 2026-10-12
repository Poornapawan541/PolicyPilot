import numpy as np


class VectorStore:
    """
    Lightweight, deterministic vector store for semantic search using NumPy.
    Replaces heavy compiled FAISS dependencies with pure NumPy matrix operations,
    guaranteeing mathematical equivalence to FAISS IndexFlatIP (cosine similarity
    on normalized vectors) with zero external C++ runtime dependencies.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype="float32")

    def add_embeddings(self, embeddings: list[list[float]]) -> None:
        """
        Add embedding vectors to the vector store.
        """
        if not embeddings:
            return

        new_vectors = np.asarray(embeddings, dtype="float32")

        if new_vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D list.")

        if new_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {new_vectors.shape[1]}."
            )

        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the most similar vectors using inner product.

        Returns:
            scores and vector positions (as float32 and int64 1D ndarrays),
            matching FAISS IndexFlatIP return signatures.
        """
        query_vector = np.asarray(query_embedding, dtype="float32")

        if query_vector.ndim == 1:
            if query_vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Expected query dimension {self.dimension}, "
                    f"got {query_vector.shape[0]}."
                )
        elif query_vector.ndim == 2:
            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    f"Expected query dimension {self.dimension}, "
                    f"got {query_vector.shape[1]}."
                )
            query_vector = query_vector[0]
        else:
            raise ValueError("Query embedding must be 1D or 2D.")

        n_total = self.vectors.shape[0]
        if n_total == 0 or top_k <= 0:
            # Handle empty index or zero top_k cleanly matching FAISS padding conventions
            pad_len = max(0, top_k)
            return (
                np.full(pad_len, -np.inf, dtype="float32"),
                np.full(pad_len, -1, dtype="int64"),
            )

        # Inner product between all stored vectors and the query vector
        raw_scores = np.dot(self.vectors, query_vector)

        # Deterministic sorting: sort descending by score, breaking ties by ascending original index
        # Using lexsort with (-raw_scores) as primary key and (original indices) as secondary key
        indices_range = np.arange(n_total, dtype="int64")
        sorted_indices = np.lexsort((indices_range, -raw_scores))

        effective_k = min(top_k, n_total)
        chosen_indices = sorted_indices[:effective_k]
        chosen_scores = raw_scores[chosen_indices].astype("float32")

        # If requested top_k is larger than stored vectors, pad with (-inf, -1) identical to FAISS
        if top_k > n_total:
            padding_count = top_k - n_total
            padded_scores = np.concatenate([chosen_scores, np.full(padding_count, -np.inf, dtype="float32")])
            padded_indices = np.concatenate([chosen_indices, np.full(padding_count, -1, dtype="int64")])
            return padded_scores, padded_indices

        return chosen_scores, chosen_indices

    def count(self) -> int:
        """
        Return the number of vectors stored.
        """
        return int(self.vectors.shape[0])