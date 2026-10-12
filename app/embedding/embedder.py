import os
from fastembed import TextEmbedding


class EmbeddingModel:
    """
    Lightweight ONNX-powered embedding model using FastEmbed.
    Generates 384-dimensional normalized embeddings with low memory footprint,
    eliminating the heavy PyTorch runtime dependency.

    Implements process-level model caching to guarantee that multiple instantiations
    reuse the exact same underlying ONNX session without duplicating memory.
    """

    _cached_model = None
    _cached_model_name = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the FastEmbed ONNX embedding model with process-level session reuse.
        """
        self.model_name = model_name
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        if EmbeddingModel._cached_model is None or EmbeddingModel._cached_model_name != model_name:
            EmbeddingModel._cached_model = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=1,
            )
            EmbeddingModel._cached_model_name = model_name
        self.model = EmbeddingModel._cached_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generates normalized vector embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, where each vector is a list of floats.
        """
        if not texts:
            return []
        embeddings = list(self.model.embed(texts))
        return [vec.tolist() for vec in embeddings]

    def embed_text(self, text: str) -> list[float]:
        """
        Generates a single normalized vector embedding for a single text string.

        Args:
            text: Single input string.

        Returns:
            Embedding vector as a list of floats.
        """
        generator = self.model.embed([text])
        first_vec = next(generator)
        return first_vec.tolist()
