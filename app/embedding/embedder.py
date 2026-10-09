from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating vector embeddings
    from text using a local sentence-transformers model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the model once into memory.
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates normalized vector embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, where each vector is a list of floats.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        """
        Generates a single normalized vector embedding for a single text string.

        Args:
            text: Single input string.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
