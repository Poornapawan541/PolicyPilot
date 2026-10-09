"""
Service loader and singleton pipeline holder for PolicyPilot FastAPI backend.

Pre-indexes documents on application startup (or provides lazy/mock injection),
reusing heavy embedding and reranker instances across requests.
"""

from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PipelineManager:
    """
    Manages the compiled PolicyPilot LangGraph workflow instance,
    reusing heavy embedding models and retrievers across all requests.
    """

    def __init__(self):
        self._graph: Optional[Any] = None
        self._corpus_chunks: Optional[list[dict]] = None
        self._initialized: bool = False

    def is_initialized(self) -> bool:
        return self._initialized and self._graph is not None

    def set_graph(self, graph: Any) -> None:
        """Allows injecting a mock or pre-compiled graph for testing or custom setups."""
        self._graph = graph
        self._initialized = True

    def initialize(self, force_reload: bool = False) -> Any:
        """
        Initializes the production graph using the default document corpus.
        Does not reload if already initialized unless force_reload=True.
        """
        if self._initialized and self._graph is not None and not force_reload:
            return self._graph

        from app.ingestion.pdf_extractor import extract_text_from_pdf
        from app.ingestion.chunker import chunk_pages
        from app.embedding.embedder import EmbeddingModel
        from app.retrieval.vector_store import VectorStore
        from app.retrieval.retriever import SemanticRetriever
        from app.retrieval.bm25_retriever import BM25Retriever
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.retrieval.reranker import CrossEncoderReranker
        from app.agents.retrieval_agent import RetrievalAgent
        from app.agents.evidence_verification_agent import EvidenceVerificationAgent
        from app.agents.reasoning_agent import ReasoningAgent
        from app.agents.response_agent import ResponseAgent
        from app.workflow.policypilot_graph import build_policypilot_graph

        doc1_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
        doc2_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"

        chunks: list[dict] = []
        if doc1_path.exists():
            pages1 = extract_text_from_pdf(
                str(doc1_path),
                document_id="PP-TEST-001",
                document_name="PolicyPilot_Test_Enterprise_Handbook.pdf",
            )
            chunks.extend(chunk_pages(pages1, chunk_size=200, overlap=30))

        if doc2_path.exists():
            pages2 = extract_text_from_pdf(
                str(doc2_path),
                document_id="PP-TEST-002",
                document_name="PolicyPilot_Test_Remote_Work_Addendum.pdf",
            )
            chunks.extend(chunk_pages(pages2, chunk_size=200, overlap=30))

        embedder = EmbeddingModel()
        texts = [c["text"] for c in chunks] if chunks else []
        embeddings = embedder.embed_texts(texts) if texts else []
        vector_store = VectorStore(dimension=384)
        if embeddings:
            vector_store.add_embeddings(embeddings)

        semantic_retriever = SemanticRetriever(vector_store)
        bm25_retriever = BM25Retriever(chunks if chunks else [{"chunk_id": "c0", "page_number": 1, "text": "empty"}])
        hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
        reranker = CrossEncoderReranker()

        retrieval_agent = RetrievalAgent(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            embedding_model=embedder,
        )
        verification_agent = EvidenceVerificationAgent()
        reasoning_agent = ReasoningAgent()
        response_agent = ResponseAgent()

        self._graph = build_policypilot_graph(
            chunks=chunks,
            retrieval_agent=retrieval_agent,
            verification_agent=verification_agent,
            reasoning_agent=reasoning_agent,
            response_agent=response_agent,
        )
        self._corpus_chunks = chunks
        self._initialized = True
        return self._graph

    def get_graph(self) -> Any:
        if not self._initialized or self._graph is None:
            return self.initialize()
        return self._graph


# Global singleton instance
pipeline_manager = PipelineManager()
