import pytest
from pathlib import Path

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent
from app.workflow.policypilot_graph import build_policypilot_graph


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def handbook_pdf_path(project_root: Path) -> Path:
    pdf_path = project_root / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Handbook PDF not found at {pdf_path}")
    return pdf_path


@pytest.fixture(scope="session")
def addendum_pdf_path(project_root: Path) -> Path:
    pdf_path = project_root / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Addendum PDF not found at {pdf_path}")
    return pdf_path


@pytest.fixture(scope="session")
def chunks(handbook_pdf_path: Path) -> list[dict]:
    pages = extract_text_from_pdf(str(handbook_pdf_path))
    return chunk_pages(pages)


@pytest.fixture(scope="session")
def all_chunks(handbook_pdf_path: Path, addendum_pdf_path: Path) -> list[dict]:
    pages1 = extract_text_from_pdf(
        str(handbook_pdf_path),
        document_id="PP-TEST-001",
        document_name="PolicyPilot_Test_Enterprise_Handbook.pdf",
    )
    chunks1 = chunk_pages(pages1, chunk_size=200, overlap=30)

    pages2 = extract_text_from_pdf(
        str(addendum_pdf_path),
        document_id="PP-TEST-002",
        document_name="PolicyPilot_Test_Remote_Work_Addendum.pdf",
    )
    chunks2 = chunk_pages(pages2, chunk_size=200, overlap=30)

    return chunks1 + chunks2


@pytest.fixture(scope="session")
def embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@pytest.fixture(scope="session")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.fixture(scope="session")
def query_agent() -> QueryUnderstandingAgent:
    return QueryUnderstandingAgent(model_name="llama3:latest")


@pytest.fixture(scope="session")
def agent(query_agent: QueryUnderstandingAgent) -> QueryUnderstandingAgent:
    """Alias for query_agent expected by test_query_understanding_agent.py."""
    return query_agent


@pytest.fixture(scope="session")
def verification_agent() -> EvidenceVerificationAgent:
    return EvidenceVerificationAgent()


@pytest.fixture(scope="session")
def reasoning_agent() -> ReasoningAgent:
    return ReasoningAgent()


@pytest.fixture(scope="session")
def response_agent() -> ResponseAgent:
    return ResponseAgent()


@pytest.fixture(scope="session")
def retrieval_agent(
    chunks: list[dict],
    embedding_model: EmbeddingModel,
    reranker: CrossEncoderReranker,
) -> RetrievalAgent:
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)

    return RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        embedding_model=embedding_model,
    )


@pytest.fixture(scope="session")
def graph(
    all_chunks: list[dict],
    embedding_model: EmbeddingModel,
    reranker: CrossEncoderReranker,
    verification_agent: EvidenceVerificationAgent,
    reasoning_agent: ReasoningAgent,
    response_agent: ResponseAgent,
):
    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(all_chunks)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)

    multi_doc_retrieval_agent = RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        embedding_model=embedding_model,
    )

    return build_policypilot_graph(
        chunks=all_chunks,
        retrieval_agent=multi_doc_retrieval_agent,
        verification_agent=verification_agent,
        reasoning_agent=reasoning_agent,
        response_agent=response_agent,
    )
