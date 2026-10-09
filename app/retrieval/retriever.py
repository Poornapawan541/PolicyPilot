from app.retrieval.vector_store import VectorStore


class SemanticRetriever:
    """
    Retrieves the most relevant document chunks using vector similarity.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query_embedding: list[float],
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Returns structured results containing:
        - chunk_id
        - page_number
        - text
        - similarity_score
        """

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for score, index in zip(scores, indices):
            if index < 0:
                continue

            chunk = chunks[int(index)]

            res = {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "similarity_score": float(score),
            }
            if "document_id" in chunk:
                res["document_id"] = chunk["document_id"]
            if "document_name" in chunk:
                res["document_name"] = chunk["document_name"]

            results.append(res)

        return results