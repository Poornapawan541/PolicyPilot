from app.embedding.embedder import EmbeddingModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker


class RetrievalAgent:
    """
    Executes structured retrieval plans produced by QueryUnderstandingAgent.
    Orchestrates embedding generation, hybrid retrieval (dense + sparse), and
    cross-encoder reranking to return the most relevant evidence chunks.

    Does not answer questions, does not generate citations, and does not perform
    evidence sufficiency verification.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        embedding_model: EmbeddingModel,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.embedding_model = embedding_model

    def validate_query_plan(self, query_plan: dict) -> None:
        """
        Validates the incoming query plan schema.
        Requires 'search_query' to be a non-empty string.
        """
        if not isinstance(query_plan, dict):
            raise ValueError("Query plan must be a dictionary.")

        search_query = query_plan.get("search_query")
        if not search_query or not isinstance(search_query, str) or not search_query.strip():
            raise ValueError(
                "Query plan must contain a non-empty 'search_query' string."
            )

    def retrieve(
        self,
        query_plan: dict,
        chunks: list[dict],
        top_k: int = 3,
        candidate_k: int = 5,
    ) -> list[dict]:
        """
        Executes the retrieval plan:
        1. Validates query plan.
        2. Safely handles empty chunks.
        3. Generates embedding for search_query.
        4. Runs hybrid retrieval (FAISS + BM25).
        5. Reranks candidates using CrossEncoder.
        6. Returns final evidence chunks.
        """
        self.validate_query_plan(query_plan)

        if not chunks:
            return []

        search_query = query_plan["search_query"].strip()

        # 1. Embed search_query for dense retrieval
        query_embedding = self.embedding_model.embed_text(search_query)

        # 2. Hybrid retrieval (FAISS + BM25)
        candidates = self.hybrid_retriever.retrieve(
            query=search_query,
            query_embedding=query_embedding,
            chunks=chunks,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        if not candidates:
            return []

        # 3. Cross-encoder reranking
        reranked_evidence = self.reranker.rerank(
            query=search_query,
            candidates=candidates,
            top_k=top_k,
        )

        return reranked_evidence
