import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def load_corpus() -> list[dict]:
    doc1_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    doc2_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"

    pages1 = extract_text_from_pdf(
        str(doc1_path),
        document_id="PP-TEST-001",
        document_name="PolicyPilot_Test_Enterprise_Handbook.pdf",
    )
    chunks1 = chunk_pages(pages1, chunk_size=200, overlap=30)

    pages2 = extract_text_from_pdf(
        str(doc2_path),
        document_id="PP-TEST-002",
        document_name="PolicyPilot_Test_Remote_Work_Addendum.pdf",
    )
    chunks2 = chunk_pages(pages2, chunk_size=200, overlap=30)

    return chunks1 + chunks2


def test_1_numerical_conflict(verification_agent, reasoning_agent, response_agent):
    """
    TEST 1 — Numerical conflict:
    Document A says remote work is allowed up to 3 days per week.
    Document B says remote work is allowed up to 2 days per week.
    Expected: conflicting, with both claims and valid source metadata.
    """
    print("\n==================================================")
    print("TEST 1 — Numerical Conflict (3 vs 2 remote days)")
    print("==================================================")
    query = "How many days per week can employees work remotely?"
    evidence = [
        {
            "chunk_id": "PP-TEST-001-page-3-chunk-1",
            "document_id": "PP-TEST-001",
            "document_name": "PolicyPilot_Test_Enterprise_Handbook.pdf",
            "page_number": 3,
            "text": "Eligible employees may work remotely up to 3 days per week with manager approval.",
        },
        {
            "chunk_id": "PP-TEST-002-page-1-chunk-1",
            "document_id": "PP-TEST-002",
            "document_name": "PolicyPilot_Test_Remote_Work_Addendum.pdf",
            "page_number": 1,
            "text": "Eligible employees may work remotely up to 2 days per week with manager approval.",
        },
    ]

    ver_res = verification_agent.verify(query, evidence)
    print("Verification result:\n", json.dumps(ver_res, indent=2))

    assert ver_res["status"] == "conflicting"
    assert ver_res["is_sufficient"] is False
    assert ver_res["supporting_evidence"] == []
    assert "conflicts" in ver_res and len(ver_res["conflicts"]) > 0

    first_conflict = ver_res["conflicts"][0]
    assert "remote" in first_conflict["topic"].lower()
    claims = first_conflict["claims"]
    assert len(claims) >= 2

    # Verify claim metadata
    doc_ids = {c.get("document_id") for c in claims}
    assert "PP-TEST-001" in doc_ids
    assert "PP-TEST-002" in doc_ids
    assert any("3 days" in c["claim"] for c in claims)
    assert any("2 days" in c["claim"] for c in claims)

    # Reasoning Gate
    reas_res = reasoning_agent.reason(query, ver_res, evidence)
    assert reas_res["status"] == "conflicting"
    assert reas_res["conclusion"] is None
    assert reas_res["evidence_used"] == []

    # Response Gate
    resp_res = response_agent.respond(query, reas_res, evidence)
    print("Response result:\n", json.dumps(resp_res, indent=2))
    assert resp_res["status"] == "conflicting"
    assert "conflict" in resp_res["answer"].lower()
    assert "PolicyPilot Test Enterprise Handbook" in resp_res["answer"]
    assert "PolicyPilot Test Remote Work Addendum" in resp_res["answer"]
    assert resp_res["citations"] == []
    print(">>> TEST 1 PASSED")


def test_2_monetary_conflict(verification_agent, reasoning_agent, response_agent):
    """
    TEST 2 — Monetary conflict:
    Two documents give different limits for the same expense category.
    Document A: Hotel reimbursement is limited to INR 4,500 per night.
    Document B: Hotel reimbursement is limited to INR 6,000 per night.
    Expected: conflicting, with both sources retained.
    """
    print("\n==================================================")
    print("TEST 2 — Monetary Conflict (Hotel Limit: INR 4,500 vs INR 6,000)")
    print("==================================================")
    query = "What is the domestic hotel reimbursement limit per night?"
    evidence = [
        {
            "chunk_id": "PP-TEST-001-page-4-chunk-1",
            "document_id": "PP-TEST-001",
            "document_name": "PolicyPilot_Test_Enterprise_Handbook.pdf",
            "page_number": 4,
            "text": "Hotel reimbursement is limited to INR 4,500 per night for domestic travel unless approved by department head.",
        },
        {
            "chunk_id": "PP-TEST-003-page-2-chunk-1",
            "document_id": "PP-TEST-003",
            "document_name": "PolicyPilot_Test_Travel_Policy_Update.pdf",
            "page_number": 2,
            "text": "Domestic hotel reimbursement is limited to INR 6,000 per night for all employees.",
        },
    ]

    ver_res = verification_agent.verify(query, evidence)
    print("Verification result:\n", json.dumps(ver_res, indent=2))

    assert ver_res["status"] == "conflicting"
    assert ver_res["is_sufficient"] is False
    assert "conflicts" in ver_res and len(ver_res["conflicts"]) > 0

    first_conflict = ver_res["conflicts"][0]
    assert "hotel" in first_conflict["topic"].lower()
    claims = first_conflict["claims"]
    assert len(claims) >= 2

    amounts = [c["claim"] for c in claims]
    assert any("4,500" in a or "4500" in a for a in amounts)
    assert any("6,000" in a or "6000" in a for a in amounts)

    reas_res = reasoning_agent.reason(query, ver_res, evidence)
    assert reas_res["status"] == "conflicting"
    assert reas_res["conclusion"] is None

    resp_res = response_agent.respond(query, reas_res, evidence)
    print("Response result:\n", json.dumps(resp_res, indent=2))
    assert resp_res["status"] == "conflicting"
    assert "conflict" in resp_res["answer"].lower()
    assert "4,500" in resp_res["answer"] or "4500" in resp_res["answer"]
    assert "6,000" in resp_res["answer"] or "6000" in resp_res["answer"]
    assert resp_res["citations"] == []
    print(">>> TEST 2 PASSED")


def test_3_complementary_evidence_not_conflict(verification_agent, reasoning_agent, response_agent):
    """
    TEST 3 — Complementary evidence:
    One document specifies remote work days (3 days); another specifies home office stipend (INR 15,000)
    and manager approval.
    Expected: sufficient/grounded, NOT conflicting.
    """
    print("\n==================================================")
    print("TEST 3 — Complementary Evidence (Not Conflicting)")
    print("==================================================")
    query = "What are the remote work rules regarding allowed days and stipend?"
    evidence = [
        {
            "chunk_id": "PP-TEST-001-page-3-chunk-1",
            "document_id": "PP-TEST-001",
            "document_name": "PolicyPilot_Test_Enterprise_Handbook.pdf",
            "page_number": 3,
            "text": "Eligible employees may work remotely up to 3 days per week with manager approval.",
        },
        {
            "chunk_id": "PP-TEST-002-page-1-chunk-1",
            "document_id": "PP-TEST-002",
            "document_name": "PolicyPilot_Test_Remote_Work_Addendum.pdf",
            "page_number": 1,
            "text": "Employees authorized for remote work are eligible for a one-time home office setup stipend of INR 15,000.",
        },
    ]

    ver_res = verification_agent.verify(query, evidence)
    print("Verification result:\n", json.dumps(ver_res, indent=2))

    assert ver_res["status"] == "sufficient"
    assert ver_res["is_sufficient"] is True
    assert len(ver_res["supporting_evidence"]) >= 1

    reas_res = reasoning_agent.reason(query, ver_res, evidence)
    assert reas_res["status"] == "grounded"
    assert reas_res["conclusion"] is not None

    resp_res = response_agent.respond(query, reas_res, evidence)
    assert resp_res["status"] == "grounded"
    assert len(resp_res["citations"]) >= 1
    print(">>> TEST 3 PASSED")


def test_4_explicit_supersession(verification_agent, reasoning_agent, response_agent):
    """
    TEST 4 — Explicit supersession:
    Document B explicitly states: "This document supersedes all prior remote work policies."
    Expected: resolved / not conflicting when explicit precedence is stated.
    """
    print("\n==================================================")
    print("TEST 4 — Explicit Supersession")
    print("==================================================")
    query = "How many days per week can employees work remotely under the updated policy?"
    evidence = [
        {
            "chunk_id": "PP-TEST-001-page-3-chunk-1",
            "document_id": "PP-TEST-001",
            "document_name": "PolicyPilot_Test_Enterprise_Handbook.pdf",
            "page_number": 3,
            "text": "Eligible employees may work remotely up to 3 days per week with manager approval.",
        },
        {
            "chunk_id": "PP-TEST-004-page-1-chunk-1",
            "document_id": "PP-TEST-004",
            "document_name": "PolicyPilot_Test_Remote_Policy_2026.pdf",
            "page_number": 1,
            "text": "This document supersedes all prior remote work policies. Employees may work remotely up to 4 days per week.",
        },
    ]

    ver_res = verification_agent.verify(query, evidence)
    print("Verification result:\n", json.dumps(ver_res, indent=2))

    # Because explicit supersession statement exists, it should not be blocked by the blind conflict guardrail
    assert ver_res["status"] == "sufficient"
    assert ver_res["is_sufficient"] is True
    print(">>> TEST 4 PASSED")


def test_5_end_to_end_graph_conflict_flow(all_chunks):
    """
    TEST 5 — End-to-end Graph Conflict Flow:
    Injects conflicting chunk into the graph and verifies routing:
    START -> query_understanding -> retrieval -> verification -> response (bypassing reasoning) -> END
    """
    print("\n==================================================")
    print("TEST 5 — End-to-end Graph Conflict Flow")
    print("==================================================")
    # Add a conflicting chunk to chunks corpus
    conflicting_chunk = {
        "chunk_id": "PP-TEST-002-page-1-conflict",
        "document_id": "PP-TEST-002",
        "document_name": "PolicyPilot_Test_Remote_Work_Addendum.pdf",
        "page_number": 1,
        "text": "Eligible employees may work remotely up to 2 days per week with direct manager approval.",
    }
    corpus_with_conflict = all_chunks + [conflicting_chunk]

    embedder = EmbeddingModel()
    texts = [c["text"] for c in corpus_with_conflict]
    embeddings = embedder.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(corpus_with_conflict)
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

    graph = build_policypilot_graph(
        chunks=corpus_with_conflict,
        retrieval_agent=retrieval_agent,
        verification_agent=verification_agent,
        reasoning_agent=reasoning_agent,
        response_agent=response_agent,
    )

    query = "How many days per week are employees allowed to work remotely?"
    result = graph.invoke({"original_query": query, "evidence": []})
    print("Graph execution result:")
    print("Status:", result.get("status"))
    print("Answer:", result.get("final_answer"))
    print("Reasoning Result:", result.get("reasoning_result"))

    assert result.get("status") == "conflicting"
    assert result.get("reasoning_result") is None  # Reasoning node was bypassed
    assert "conflict" in result.get("final_answer", "").lower()
    assert result.get("response_result", {}).get("citations") == []
    print(">>> TEST 5 PASSED")


def main():
    print("==================================================")
    print("RUNNING STEP 19 CROSS-DOCUMENT CONFLICT DETECTION TESTS")
    print("==================================================")

    verification_agent = EvidenceVerificationAgent()
    reasoning_agent = ReasoningAgent()
    response_agent = ResponseAgent()
    all_chunks = load_corpus()

    test_1_numerical_conflict(verification_agent, reasoning_agent, response_agent)
    test_2_monetary_conflict(verification_agent, reasoning_agent, response_agent)
    test_3_complementary_evidence_not_conflict(verification_agent, reasoning_agent, response_agent)
    test_4_explicit_supersession(verification_agent, reasoning_agent, response_agent)
    test_5_end_to_end_graph_conflict_flow(all_chunks)

    print("\n==================================================")
    print("ALL STEP 19 CONFLICT DETECTION TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
