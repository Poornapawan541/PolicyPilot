import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent
from app.workflow.policypilot_graph import (
    build_policypilot_graph,
    route_after_verification,
    PolicyPilotState,
)


def test_graph_structure_and_delegation():
    print("==================================================")
    print("RUNNING ARCHITECTURE & STRUCTURE TESTS")
    print("==================================================")

    dummy_chunks = [{"chunk_id": "c1", "page_number": 1, "text": "Test text"}]
    mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
    mock_retrieval_agent = MagicMock(spec=RetrievalAgent)
    mock_verification_agent = MagicMock(spec=EvidenceVerificationAgent)
    mock_reasoning_agent = MagicMock(spec=ReasoningAgent)
    mock_response_agent = MagicMock(spec=ResponseAgent)

    graph = build_policypilot_graph(
        chunks=dummy_chunks,
        query_agent=mock_query_agent,
        retrieval_agent=mock_retrieval_agent,
        verification_agent=mock_verification_agent,
        reasoning_agent=mock_reasoning_agent,
        response_agent=mock_response_agent,
    )

    # TEST 1: Graph Structure & Nodes
    expected_nodes = {"query_understanding", "retrieval", "verification", "reasoning", "response"}
    graph_nodes = set(graph.nodes.keys())
    for node in expected_nodes:
        assert node in graph_nodes, f"Node {node} missing from graph nodes: {graph_nodes}"
    print(f"Test 1 (Graph construction and expected nodes): Passed. Nodes: {graph_nodes}")

    # TEST 1 (b): Routing logic function verification
    assert route_after_verification({"verification_result": {"status": "sufficient"}}) == "reasoning"
    assert route_after_verification({"verification_result": {"status": "insufficient"}}) == "response"
    assert route_after_verification({"verification_result": {"status": "conflicting"}}) == "response"
    assert route_after_verification({"error": "some error"}) == "response"
    assert route_after_verification({}) == "response"
    print("Test 1b (Routing function verification): Passed")

    # TEST 7 & ARCHITECTURE TEST: Insufficient Evidence Routing & Reasoning NOT Called
    print("\n--- Test 7: Verify Reasoning is NOT Called on Insufficient Evidence ---")
    mock_query_agent.analyze.return_value = {
        "original_query": "What is the parental leave duration?",
        "intent": "parental leave",
        "search_query": "parental leave",
        "keywords": ["parental leave"],
        "filters": {},
    }
    mock_retrieval_agent.retrieve.return_value = [
        {"chunk_id": "c1", "page_number": 2, "text": "Annual leave is 24 days."}
    ]
    mock_verification_agent.verify.return_value = {
        "original_query": "What is the parental leave duration?",
        "status": "insufficient",
        "is_sufficient": False,
        "confidence": 1.0,
        "reason": "Not mentioned",
        "supporting_evidence": [],
    }
    # Reasoning agent must raise if erroneously called!
    mock_reasoning_agent.reason.side_effect = AssertionError("ReasoningAgent was called when evidence was insufficient!")

    mock_response_agent.respond.return_value = {
        "original_query": "What is the parental leave duration?",
        "status": "insufficient",
        "answer": "Insufficient evidence in the provided documents to answer this question.",
        "citations": [],
    }

    res_insuf = graph.invoke({
        "original_query": "What is the parental leave duration?",
    })

    assert res_insuf["status"] == "insufficient"
    assert "Insufficient evidence" in res_insuf["final_answer"]
    assert res_insuf["reasoning_result"] is None
    mock_reasoning_agent.reason.assert_not_called()
    mock_response_agent.respond.assert_called_once()
    print("Test 7 (Insufficient evidence bypassed reasoning to response): Passed")

    # TEST 6 & 7: Conflicting Evidence Routing & Reasoning NOT Called
    print("\n--- Test 6: Verify Reasoning is NOT Called on Conflicting Evidence ---")
    mock_reasoning_agent.reason.reset_mock()
    mock_response_agent.respond.reset_mock()
    mock_reasoning_agent.reason.side_effect = AssertionError("ReasoningAgent was called when evidence was conflicting!")

    mock_verification_agent.verify.return_value = {
        "original_query": "How many days of leave?",
        "status": "conflicting",
        "is_sufficient": False,
        "confidence": 1.0,
        "reason": "Contradictory information",
        "supporting_evidence": [],
    }
    mock_response_agent.respond.return_value = {
        "original_query": "How many days of leave?",
        "status": "conflicting",
        "answer": "The provided documents contain conflicting information, so a reliable answer cannot be determined.",
        "citations": [],
    }

    res_conf = graph.invoke({
        "original_query": "How many days of leave?",
    })

    assert res_conf["status"] == "conflicting"
    assert "conflicting information" in res_conf["final_answer"]
    assert res_conf["reasoning_result"] is None
    mock_reasoning_agent.reason.assert_not_called()
    mock_response_agent.respond.assert_called_once()
    print("Test 6 (Conflicting evidence bypassed reasoning to response): Passed")

    # TEST 8 & 9: Sufficient Evidence Calls Reasoning then Response
    print("\n--- Test 8 & 9: Verify Response and Reasoning are Called on Sufficient ---")
    mock_reasoning_agent.reason.reset_mock()
    mock_response_agent.respond.reset_mock()
    mock_reasoning_agent.reason.side_effect = None
    mock_reasoning_agent.reason.return_value = {
        "original_query": "How many days of leave?",
        "status": "grounded",
        "conclusion": "24 days",
        "reason": "Policy says 24 days",
        "evidence_used": [{"chunk_id": "c1", "page_number": 2}],
    }
    mock_verification_agent.verify.return_value = {
        "original_query": "How many days of leave?",
        "status": "sufficient",
        "is_sufficient": True,
        "confidence": 0.95,
        "reason": "Direct support",
        "supporting_evidence": [{"chunk_id": "c1", "page_number": 2}],
    }
    mock_response_agent.respond.return_value = {
        "original_query": "How many days of leave?",
        "status": "grounded",
        "answer": "Employees receive 24 days of annual leave per calendar year. [Page 2]",
        "citations": [{"chunk_id": "c1", "page_number": 2, "citation": "[Page 2]"}],
    }

    res_suff = graph.invoke({"original_query": "How many days of leave?"})
    mock_reasoning_agent.reason.assert_called_once()
    mock_response_agent.respond.assert_called_once()
    assert res_suff["status"] == "grounded"
    assert res_suff["reasoning_result"] is not None
    assert res_suff["response_result"] is not None
    assert res_suff["final_answer"] == "Employees receive 24 days of annual leave per calendar year. [Page 2]"
    print("Test 8 & 9 Passed.")

    # TEST 10: Error Propagation
    print("\n--- Test 10: Error Propagation ---")
    mock_query_agent.analyze.side_effect = RuntimeError("Query analysis crash")
    res_err = graph.invoke({"original_query": "Crashing query"})
    assert res_err["status"] == "error"
    assert "Query analysis crash" in res_err["error"]
    assert res_err["final_answer"] is not None
    mock_query_agent.analyze.side_effect = None
    print("Test 10 (Safe error propagation): Passed")

    # TEST 11: Invalid Empty Query
    print("\n--- Test 11: Invalid Empty Query ---")
    res_empty_q = graph.invoke({"original_query": ""})
    assert res_empty_q["status"] == "error"
    assert "Query cannot be empty" in res_empty_q["final_answer"]
    print("Test 11 Passed.")

    # TEST 12: Multiple Invocations State Isolation
    print("\n--- Test 12: State Isolation Across Multiple Invocations ---")
    inv1 = graph.invoke({"original_query": "How many days of leave?"})
    inv2 = graph.invoke({"original_query": ""})
    assert inv1["status"] == "grounded"
    assert inv2["status"] == "error"
    assert inv1["error"] is None
    assert inv2["error"] is not None
    print("Test 12 (State isolation across invocations): Passed\n")


def run_live_end_to_end_tests(graph):
    print("==================================================")
    print("RUNNING LIVE END-TO-END WORKFLOW TESTS")
    print("==================================================")

    # -------------------------------------------------------------
    # TEST 2: Annual Leave Question End-to-End
    # -------------------------------------------------------------
    print("\n--- Test 2: Annual Leave End-to-End ---")
    q1 = "How many days of annual leave do employees receive?"
    res1 = graph.invoke({"original_query": q1})
    print(f"Final Answer: {res1.get('final_answer')}")
    print(f"Status: {res1.get('status')}")
    assert res1.get("status") == "grounded"
    assert res1.get("final_answer") is not None
    assert "24" in res1["final_answer"]
    assert "[Page 2]" in res1["final_answer"]
    assert res1.get("reasoning_result") is not None
    assert res1.get("error") is None
    print("Test 2 Passed.")

    # -------------------------------------------------------------
    # TEST 3: Remote Work Question End-to-End
    # -------------------------------------------------------------
    print("\n--- Test 3: Remote Work End-to-End ---")
    q2 = "Can employees work remotely three days per week?"
    res2 = graph.invoke({"original_query": q2})
    print(f"Final Answer: {res2.get('final_answer')}")
    print(f"Status: {res2.get('status')}")
    assert res2.get("status") == "grounded"
    assert res2.get("final_answer") is not None
    assert "3 days" in res2["final_answer"].lower() or "three days" in res2["final_answer"].lower()
    assert "approval" in res2["final_answer"].lower()
    assert "[Page 3]" in res2["final_answer"]
    assert res2.get("reasoning_result") is not None
    assert res2.get("error") is None
    print("Test 3 Passed.")

    # -------------------------------------------------------------
    # TEST 4: Password Policy Question End-to-End
    # -------------------------------------------------------------
    print("\n--- Test 4: Password Policy End-to-End ---")
    q3 = "What is the password policy?"
    res3 = graph.invoke({"original_query": q3})
    print(f"Final Answer: {res3.get('final_answer')}")
    print(f"Status: {res3.get('status')}")
    assert res3.get("status") == "grounded"
    assert res3.get("final_answer") is not None
    assert "12" in res3["final_answer"] or "password" in res3["final_answer"].lower()
    assert "[Page 5]" in res3["final_answer"]
    assert res3.get("reasoning_result") is not None
    assert res3.get("error") is None
    print("Test 4 Passed.")

    # -------------------------------------------------------------
    # TEST 5: Parental Leave (Insufficient Evidence) End-to-End
    # -------------------------------------------------------------
    print("\n--- Test 5: Parental Leave (Insufficient Evidence) End-to-End ---")
    q4 = "What is the parental leave duration?"
    res4 = graph.invoke({"original_query": q4})
    print(f"Final Answer: {res4.get('final_answer')}")
    print(f"Status: {res4.get('status')}")
    assert res4.get("status") == "insufficient"
    assert "Insufficient evidence" in res4.get("final_answer", "")
    assert res4.get("reasoning_result") is None, "Reasoning ran on insufficient evidence!"
    assert res4.get("error") is None
    print("Test 5 Passed.")


def main():
    test_graph_structure_and_delegation()

    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    print(f"Loaded {len(pages)} pages and {len(chunks)} chunks.")

    print("Building full live PolicyPilot LangGraph...")
    from app.retrieval.vector_store import VectorStore
    from app.retrieval.retriever import SemanticRetriever
    from app.retrieval.bm25_retriever import BM25Retriever

    embedder = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    reranker = CrossEncoderReranker()
    retrieval_agent = RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        embedding_model=embedder,
    )

    query_agent = QueryUnderstandingAgent()
    verification_agent = EvidenceVerificationAgent()
    reasoning_agent = ReasoningAgent()
    response_agent = ResponseAgent()

    graph = build_policypilot_graph(
        chunks=chunks,
        query_agent=query_agent,
        retrieval_agent=retrieval_agent,
        verification_agent=verification_agent,
        reasoning_agent=reasoning_agent,
        response_agent=response_agent,
    )

    run_live_end_to_end_tests(graph)

    print("\n==================================================")
    print("ALL POLICYPILOT STEP 16 LANGGRAPH TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
