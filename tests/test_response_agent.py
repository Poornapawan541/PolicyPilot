import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent


def run_unit_tests(agent: ResponseAgent):
    print("==================================================")
    print("RUNNING UNIT TESTS (EDGE CASES & VALIDATION)")
    print("==================================================")

    sample_evidence = [
        {
            "chunk_id": "page-2-chunk-1",
            "page_number": 2,
            "text": "Employees receive 24 days of annual leave per calendar year.",
        },
        {
            "chunk_id": "page-3-chunk-1",
            "page_number": 3,
            "text": "Remote work allowed up to 3 days per week with manager approval.",
        },
    ]

    # -------------------------------------------------------------
    # 1. Invalid Query
    # -------------------------------------------------------------
    print("\n--- Test 12: Invalid Query Handling ---")
    try:
        agent.respond("", {"status": "grounded", "conclusion": "24 days", "evidence_used": [{"chunk_id": "page-2-chunk-1", "page_number": 2}]}, sample_evidence)
        assert False, "Should raise ValueError on empty query"
    except ValueError as e:
        print(f"Empty query caught as expected: {e}")

    try:
        agent.respond(None, {"status": "grounded"}, sample_evidence)
        assert False, "Should raise ValueError on None query"
    except ValueError as e:
        print(f"None query caught as expected: {e}")

    # -------------------------------------------------------------
    # 2. Insufficient Evidence Gate
    # -------------------------------------------------------------
    print("\n--- Test 6: Insufficient Evidence Gate ---")
    res_insuf = agent.respond(
        "What is parental leave duration?",
        {
            "status": "insufficient",
            "conclusion": None,
            "evidence_used": [],
        },
        sample_evidence,
    )
    print(json.dumps(res_insuf, indent=2))
    assert res_insuf["status"] == "insufficient"
    assert res_insuf["answer"] == ResponseAgent.INSUFFICIENT_ANSWER
    assert res_insuf["citations"] == []
    print("Test 6 Passed.")

    # -------------------------------------------------------------
    # 3. Conflicting Evidence Gate
    # -------------------------------------------------------------
    print("\n--- Test 7: Conflicting Evidence Gate ---")
    res_conf = agent.respond(
        "How many days of annual leave do employees receive?",
        {
            "status": "conflicting",
            "conclusion": None,
            "evidence_used": [],
        },
        sample_evidence,
    )
    print(json.dumps(res_conf, indent=2))
    assert res_conf["status"] == "conflicting"
    assert res_conf["answer"] == ResponseAgent.CONFLICTING_ANSWER
    assert res_conf["citations"] == []
    print("Test 7 Passed.")

    # -------------------------------------------------------------
    # 4. Empty Evidence Handling
    # -------------------------------------------------------------
    print("\n--- Test 8: Empty Evidence Handling ---")
    res_empty = agent.respond(
        "How many days of leave?",
        {
            "status": "grounded",
            "conclusion": "24 days",
            "evidence_used": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
        },
        [],
    )
    print(json.dumps(res_empty, indent=2))
    assert res_empty["status"] == "insufficient"
    assert res_empty["answer"] == ResponseAgent.INSUFFICIENT_ANSWER
    assert res_empty["citations"] == []
    print("Test 8 Passed.")

    # -------------------------------------------------------------
    # 5. Unknown Evidence / Chunk ID Rejection (Gate / Safety)
    # -------------------------------------------------------------
    print("\n--- Test 9: Unknown Chunk ID Rejection ---")
    res_bad_chunk = agent.respond(
        "How many days of leave?",
        {
            "status": "grounded",
            "conclusion": "24 days",
            "evidence_used": [{"chunk_id": "page-99-chunk-1", "page_number": 99}],
        },
        sample_evidence,
    )
    print(json.dumps(res_bad_chunk, indent=2))
    assert res_bad_chunk["status"] == "insufficient"
    assert res_bad_chunk["citations"] == []
    print("Test 9 Passed.")

    # -------------------------------------------------------------
    # 6. Deterministic Citations Generation & Multiple Citations
    # -------------------------------------------------------------
    print("\n--- Test 4 & 5: Deterministic Citations & Multiple Citations ---")
    original_call = agent._call_ollama
    agent._call_ollama = lambda prompt: "Eligible employees receive annual leave and remote options."

    res_multi_cit = agent.respond(
        "What are leave and remote benefits?",
        {
            "status": "grounded",
            "conclusion": "Employees receive 24 days leave and up to 3 days remote work.",
            "evidence_used": [
                {"chunk_id": "page-2-chunk-1", "page_number": 2},
                {"chunk_id": "page-3-chunk-1", "page_number": 3},
            ],
        },
        sample_evidence,
    )
    print(json.dumps(res_multi_cit, indent=2))
    assert res_multi_cit["status"] == "grounded"
    assert len(res_multi_cit["citations"]) == 2
    assert res_multi_cit["citations"][0]["citation"] == "[Page 2]"
    assert res_multi_cit["citations"][1]["citation"] == "[Page 3]"
    assert "[Page 2]" in res_multi_cit["answer"]
    assert "[Page 3]" in res_multi_cit["answer"]
    print("Test 4 & 5 Passed.")

    # -------------------------------------------------------------
    # 7. Empty LLM Response Handling
    # -------------------------------------------------------------
    print("\n--- Test 10: Empty LLM Response Handling ---")
    agent._call_ollama = lambda prompt: ""
    res_empty_llm = agent.respond(
        "How many days of leave?",
        {
            "status": "grounded",
            "conclusion": "Employees receive 24 days of annual leave.",
            "evidence_used": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
        },
        sample_evidence,
    )
    print(json.dumps(res_empty_llm, indent=2))
    assert res_empty_llm["status"] == "grounded"
    assert "24 days" in res_empty_llm["answer"]
    assert "[Page 2]" in res_empty_llm["answer"]
    print("Test 10 Passed.")

    # -------------------------------------------------------------
    # 8. Malformed / Exception LLM Handling
    # -------------------------------------------------------------
    print("\n--- Test 11: Malformed / Error LLM Handling ---")
    def raise_err(prompt):
        raise RuntimeError("Ollama connection failed")

    agent._call_ollama = raise_err
    res_err_llm = agent.respond(
        "How many days of leave?",
        {
            "status": "grounded",
            "conclusion": "Employees receive 24 days of annual leave.",
            "evidence_used": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
        },
        sample_evidence,
    )
    print(json.dumps(res_err_llm, indent=2))
    assert res_err_llm["status"] == "grounded"
    assert "24 days" in res_err_llm["answer"]
    assert "[Page 2]" in res_err_llm["answer"]
    print("Test 11 Passed.")

    # -------------------------------------------------------------
    # 9. Verify LLM Cannot Invent / Fabricate Citations
    # -------------------------------------------------------------
    print("\n--- Test 15: Verify LLM Cannot Fabricate Citations ---")
    agent._call_ollama = lambda prompt: "Employees receive 24 days [Page 99]."
    res_fake_cit = agent.respond(
        "How many days of leave?",
        {
            "status": "grounded",
            "conclusion": "Employees receive 24 days.",
            "evidence_used": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
        },
        sample_evidence,
    )
    print(json.dumps(res_fake_cit, indent=2))
    assert "[Page 99]" not in res_fake_cit["answer"], "Fabricated [Page 99] was not stripped!"
    assert "[Page 2]" in res_fake_cit["answer"]
    print("Test 15 Passed.")

    # Restore original Ollama call
    agent._call_ollama = original_call
    print("\nAll unit tests passed successfully!")


def run_live_tests(
    verification_agent: EvidenceVerificationAgent,
    reasoning_agent: ReasoningAgent,
    response_agent: ResponseAgent,
    chunks: list[dict],
):
    print("\n==================================================")
    print("RUNNING LIVE LLM RESPONSE AGENT TESTS")
    print("==================================================")

    # -------------------------------------------------------------
    # 1. Annual Leave Grounded Response
    # -------------------------------------------------------------
    print("\n--- Live Test 1: Annual Leave Grounded Response ---")
    q1 = "How many days of annual leave do employees receive?"
    ev1 = [c for c in chunks if c.get("page_number") == 2 and "annual leave" in c.get("text", "").lower()]
    ver1 = verification_agent.verify(q1, ev1)
    reas1 = reasoning_agent.reason(q1, ver1, ev1)
    resp1 = response_agent.respond(q1, reas1, ev1)
    print(json.dumps(resp1, indent=2))

    assert resp1["status"] == "grounded"
    assert "24" in resp1["answer"]
    assert "[Page 2]" in resp1["answer"]
    assert len(resp1["citations"]) == 1
    assert resp1["citations"][0]["chunk_id"] == "page-2-chunk-1"
    assert resp1["citations"][0]["page_number"] == 2
    # Hallucination resistance test
    ans1_lower = resp1["answer"].lower()
    assert "probation" not in ans1_lower
    assert "termination" not in ans1_lower
    print("Live Test 1 Passed.")

    # -------------------------------------------------------------
    # 2. Remote Work Grounded Response
    # -------------------------------------------------------------
    print("\n--- Live Test 2: Remote Work Grounded Response ---")
    q2 = "Can employees work remotely three days per week?"
    ev2 = [c for c in chunks if c.get("page_number") == 3 and "remote" in c.get("text", "").lower()]
    ver2 = verification_agent.verify(q2, ev2)
    reas2 = reasoning_agent.reason(q2, ver2, ev2)
    resp2 = response_agent.respond(q2, reas2, ev2)
    print(json.dumps(resp2, indent=2))

    assert resp2["status"] == "grounded"
    assert "3 days" in resp2["answer"].lower() or "three days" in resp2["answer"].lower()
    assert "[Page 3]" in resp2["answer"]
    assert len(resp2["citations"]) == 1
    assert resp2["citations"][0]["chunk_id"] == "page-3-chunk-1"
    assert resp2["citations"][0]["page_number"] == 3
    print("Live Test 2 Passed.")

    # -------------------------------------------------------------
    # 3. Password Policy Grounded Response
    # -------------------------------------------------------------
    print("\n--- Live Test 3: Password Policy Grounded Response ---")
    q3 = "What is the password policy?"
    ev3 = [c for c in chunks if c.get("page_number") == 5 and "password" in c.get("text", "").lower()]
    ver3 = verification_agent.verify(q3, ev3)
    reas3 = reasoning_agent.reason(q3, ver3, ev3)
    resp3 = response_agent.respond(q3, reas3, ev3)
    print(json.dumps(resp3, indent=2))

    assert resp3["status"] == "grounded"
    assert "12" in resp3["answer"] or "password" in resp3["answer"].lower()
    assert "[Page 5]" in resp3["answer"]
    assert len(resp3["citations"]) == 1
    assert resp3["citations"][0]["chunk_id"] == "page-5-chunk-1"
    assert resp3["citations"][0]["page_number"] == 5
    print("Live Test 3 Passed.")

    # -------------------------------------------------------------
    # 4. Insufficient Evidence (Parental Leave)
    # -------------------------------------------------------------
    print("\n--- Live Test 4: Insufficient Evidence (Parental Leave) ---")
    q4 = "What is the parental leave duration?"
    ver4 = verification_agent.verify(q4, ev1)
    reas4 = reasoning_agent.reason(q4, ver4, ev1)
    resp4 = response_agent.respond(q4, reas4, ev1)
    print(json.dumps(resp4, indent=2))

    assert resp4["status"] == "insufficient"
    assert resp4["answer"] == ResponseAgent.INSUFFICIENT_ANSWER
    assert resp4["citations"] == []
    print("Live Test 4 Passed.")


def main():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    print(f"Loaded {len(pages)} pages and {len(chunks)} chunks.")

    verification_agent = EvidenceVerificationAgent()
    reasoning_agent = ReasoningAgent()
    response_agent = ResponseAgent()

    run_unit_tests(response_agent)
    run_live_tests(verification_agent, reasoning_agent, response_agent, chunks)

    print("\n==================================================")
    print("ALL RESPONSE AGENT TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
