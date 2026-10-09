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
    """
    Ingests both synthetic enterprise test documents using the existing ingestion pipeline:
    1. PolicyPilot_Test_Enterprise_Handbook.pdf (PP-TEST-001)
    2. PolicyPilot_Test_Remote_Work_Addendum.pdf (PP-TEST-002)
    """
    doc1_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    doc2_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"

    assert doc1_path.exists(), f"Document 1 missing at {doc1_path}"
    assert doc2_path.exists(), f"Document 2 missing at {doc2_path}"

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

    all_chunks = chunks1 + chunks2
    print(f"Loaded {len(chunks1)} chunks from Handbook and {len(chunks2)} chunks from Remote Work Addendum.")
    print(f"Total multi-document chunks: {len(all_chunks)}")
    return all_chunks


def setup_graph(all_chunks: list[dict]):
    """
    Initializes standard indexing and builds the compiled PolicyPilot workflow graph.
    """
    embedder = EmbeddingModel()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(all_chunks)
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
        chunks=all_chunks,
        retrieval_agent=retrieval_agent,
        verification_agent=verification_agent,
        reasoning_agent=reasoning_agent,
        response_agent=response_agent,
    )
    return graph, verification_agent, reasoning_agent, response_agent


def test_1_single_document_regression(graph):
    """
    TEST 1 — Single-document regression
    A question answerable entirely from the original handbook still works.
    Question: How many days of annual leave do employees receive per year?
    Expected: grounded answer with correct document/page citation.
    """
    print("\n==================================================")
    print("TEST 1 — Single-document regression")
    print("==================================================")
    query = "How many days of annual leave do employees receive per year?"
    result = graph.invoke({"original_query": query, "evidence": []})
    print(f"Query: {query}")
    print(f"Final status: {result.get('status')}")
    print(f"Final answer: {result.get('final_answer')}")
    print(f"Citations: {json.dumps(result.get('response_result', {}).get('citations'), indent=2)}")

    assert result.get("status") == "grounded", f"Expected grounded status, got {result.get('status')}"
    answer = result.get("final_answer", "")
    assert "24" in answer, f"Expected '24' in answer, got: {answer}"
    citations = result.get("response_result", {}).get("citations", [])
    assert len(citations) > 0, "Expected at least one citation"
    assert any(c.get("page_number") == 2 for c in citations), "Expected Page 2 citation"
    print(">>> TEST 1 PASSED")


def test_2_multi_document_complementary_evidence(graph):
    """
    TEST 2 — Multi-document complementary evidence
    Create a question whose answer requires:
    Document A: remote work allowed up to 3 days/week with manager approval.
    Document B: home office stipend / quarterly security awareness training / core availability.
    Question: What are the remote work rules and home office stipend amount?
    Expected: grounded answer combining BOTH facts with citations from both documents.
    """
    print("\n==================================================")
    print("TEST 2 — Multi-document complementary evidence")
    print("==================================================")
    query = "What are the remote work rules regarding allowed days per week and the home office stipend amount?"
    result = graph.invoke({"original_query": query, "evidence": []})
    print(f"Query: {query}")
    print(f"Final status: {result.get('status')}")
    print(f"Final answer: {result.get('final_answer')}")
    print(f"Citations: {json.dumps(result.get('response_result', {}).get('citations'), indent=2)}")

    assert result.get("status") == "grounded", f"Expected grounded status, got {result.get('status')}"
    answer = result.get("final_answer", "").lower()
    # Must combine both facts
    assert ("3 days" in answer or "three days" in answer), f"Expected 3 days from handbook in answer, got: {answer}"
    assert ("15,000" in answer or "15000" in answer), f"Expected INR 15,000 stipend from addendum in answer, got: {answer}"

    citations = result.get("response_result", {}).get("citations", [])
    assert len(citations) >= 2, f"Expected citations from both documents, got {len(citations)}"

    cited_docs = {c.get("document_id") or c.get("document_name") for c in citations}
    assert any("001" in str(d) or "Handbook" in str(d) for d in cited_docs), "Expected citation from Document 1 (Handbook)"
    assert any("002" in str(d) or "Addendum" in str(d) for d in cited_docs), "Expected citation from Document 2 (Addendum)"
    print(">>> TEST 2 PASSED")


def test_3_multi_document_specific_value(graph):
    """
    TEST 3 — Multi-document specific-value question
    Question requiring a fact from the second document that does not exist in the first.
    Question: How much is the one-time home office setup stipend for remote employees?
    Expected: correct answer (INR 15,000) with second-document citation.
    """
    print("\n==================================================")
    print("TEST 3 — Multi-document specific-value question")
    print("==================================================")
    query = "How much is the one-time home office setup stipend for remote employees?"
    result = graph.invoke({"original_query": query, "evidence": []})
    print(f"Query: {query}")
    print(f"Final status: {result.get('status')}")
    print(f"Final answer: {result.get('final_answer')}")
    print(f"Citations: {json.dumps(result.get('response_result', {}).get('citations'), indent=2)}")

    assert result.get("status") == "grounded", f"Expected grounded status, got {result.get('status')}"
    answer = result.get("final_answer", "")
    assert ("15,000" in answer or "15000" in answer), f"Expected 15,000 in answer, got: {answer}"
    citations = result.get("response_result", {}).get("citations", [])
    assert len(citations) >= 1, "Expected citation from Addendum"
    cited_docs = {c.get("document_id") or c.get("document_name") for c in citations}
    assert any("002" in str(d) or "Addendum" in str(d) for d in cited_docs), "Expected citation from Remote Work Addendum"
    print(">>> TEST 3 PASSED")


def test_4_multi_document_insufficient_evidence(graph):
    """
    TEST 4 — Multi-document insufficient evidence
    Ask a question where neither document contains the requested information.
    Question: What is the company stock option vesting schedule?
    Expected: insufficient.
    """
    print("\n==================================================")
    print("TEST 4 — Multi-document insufficient evidence")
    print("==================================================")
    query = "What is the company stock option vesting schedule?"
    result = graph.invoke({"original_query": query, "evidence": []})
    print(f"Query: {query}")
    print(f"Final status: {result.get('status')}")
    print(f"Final answer: {result.get('final_answer')}")
    print(f"Citations: {json.dumps(result.get('response_result', {}).get('citations'), indent=2)}")

    assert result.get("status") == "insufficient", f"Expected insufficient status, got {result.get('status')}"
    assert result.get("final_answer") == ResponseAgent.INSUFFICIENT_ANSWER
    citations = result.get("response_result", {}).get("citations", [])
    assert citations == [], f"Expected empty citations on insufficient evidence, got {citations}"
    print(">>> TEST 4 PASSED")


def test_5_cross_document_contradiction(verification_agent, reasoning_agent, response_agent):
    """
    TEST 5 — Cross-document contradiction
    Create conflicting evidence across the two documents.
    Doc A says 3 remote days.
    Doc B says 2 remote days.
    Question: How many days per week can employees work remotely?
    Expected: conflicting or safely insufficient. It must NOT arbitrarily choose one value.
    """
    print("\n==================================================")
    print("TEST 5 — Cross-document contradiction")
    print("==================================================")
    query = "How many days per week can employees work remotely?"
    conflicting_evidence = [
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

    ver_res = verification_agent.verify(query, conflicting_evidence)
    print("Verification result for contradiction:")
    print(json.dumps(ver_res, indent=2))

    assert ver_res.get("status") in ("conflicting", "insufficient"), (
        f"Expected conflicting or insufficient status, got {ver_res.get('status')}"
    )
    assert ver_res.get("is_sufficient") is False

    reas_res = reasoning_agent.reason(query, ver_res, conflicting_evidence)
    print("Reasoning result for contradiction:")
    print(json.dumps(reas_res, indent=2))
    assert reas_res.get("status") in ("conflicting", "insufficient")
    assert reas_res.get("conclusion") is None

    resp_res = response_agent.respond(query, reas_res, conflicting_evidence)
    print("Response result for contradiction:")
    print(json.dumps(resp_res, indent=2))
    assert resp_res.get("status") in ("conflicting", "insufficient")
    assert resp_res.get("citations") == []
    # Must NOT arbitrarily choose one number as valid answer
    assert "2 days" not in resp_res.get("answer") or "conflict" in resp_res.get("answer").lower()
    print(">>> TEST 5 PASSED")


def test_6_citation_correctness(graph):
    """
    TEST 6 — Citation correctness
    For a multi-document answer:
    - every cited document must actually contain supporting evidence
    - every page number must be correct
    - no fabricated document/page references
    """
    print("\n==================================================")
    print("TEST 6 — Citation correctness")
    print("==================================================")
    query = "What are the remote work rules regarding allowed days per week and the home office stipend amount?"
    result = graph.invoke({"original_query": query, "evidence": []})
    citations = result.get("response_result", {}).get("citations", [])
    print(f"Citations checked: {json.dumps(citations, indent=2)}")

    assert len(citations) >= 2, "Expected citations from multiple documents"

    valid_docs = {"PolicyPilot_Test_Enterprise_Handbook.pdf", "PolicyPilot_Test_Remote_Work_Addendum.pdf", "PP-TEST-001", "PP-TEST-002"}

    for cit in citations:
        doc = cit.get("document_name") or cit.get("document_id")
        assert doc in valid_docs, f"Invalid/fabricated document reference: {doc}"
        page = cit.get("page_number")
        assert isinstance(page, int), f"Page number must be int: {page}"

        if "PP-TEST-001" in str(doc) or "Handbook" in str(doc):
            assert page in (1, 2, 3, 4, 5, 6, 7), f"Handbook page out of bounds: {page}"
            assert page == 3, f"Expected page 3 for remote work handbook rule, got {page}"
        elif "PP-TEST-002" in str(doc) or "Addendum" in str(doc):
            assert page in (1, 2), f"Addendum page out of bounds: {page}"
            assert page == 1, f"Expected page 1 for stipend rule, got {page}"

        cit_tag = cit.get("citation", "")
        assert cit_tag in result.get("final_answer", ""), f"Citation tag {cit_tag} missing in final answer"

    print(">>> TEST 6 PASSED")


def main():
    print("==================================================")
    print("RUNNING STEP 18 MULTI-DOCUMENT REASONING TEST SUITE")
    print("==================================================")

    all_chunks = load_corpus()
    graph, verification_agent, reasoning_agent, response_agent = setup_graph(all_chunks)

    test_1_single_document_regression(graph)
    test_2_multi_document_complementary_evidence(graph)
    test_3_multi_document_specific_value(graph)
    test_4_multi_document_insufficient_evidence(graph)
    test_5_cross_document_contradiction(verification_agent, reasoning_agent, response_agent)
    test_6_citation_correctness(graph)

    print("\n==================================================")
    print("ALL STEP 18 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
