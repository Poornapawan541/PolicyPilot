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


def run_unit_tests(reasoning_agent: ReasoningAgent):
    print("==================================================")
    print("RUNNING UNIT TESTS (EDGE CASES & VALIDATION)")
    print("==================================================")

    sample_evidence = [
        {
            "chunk_id": "page-2-chunk-1",
            "page_number": 2,
            "text": "Employees receive 24 days of annual leave per calendar year.",
        }
    ]

    # -------------------------------------------------------------
    # TEST 10: Invalid Query
    # -------------------------------------------------------------
    print("\n--- Test 10: Invalid Query ---")
    try:
        reasoning_agent.reason(
            "",
            {"status": "sufficient", "is_sufficient": True},
            sample_evidence,
        )
        assert False, "Should raise ValueError on empty query"
    except ValueError as e:
        print(f"Empty query caught as expected: {e}")

    try:
        reasoning_agent.reason(
            None,
            {"status": "sufficient", "is_sufficient": True},
            sample_evidence,
        )
        assert False, "Should raise ValueError on None query"
    except ValueError as e:
        print(f"None query caught as expected: {e}")

    # -------------------------------------------------------------
    # TEST 9: Empty Evidence
    # -------------------------------------------------------------
    print("\n--- Test 9: Empty Evidence ---")
    res_empty = reasoning_agent.reason(
        "What is the parental leave duration?",
        {"status": "sufficient", "is_sufficient": True},
        [],
    )
    print(json.dumps(res_empty, indent=2))
    assert res_empty["status"] == "insufficient"
    assert res_empty["conclusion"] is None
    assert res_empty["evidence_used"] == []
    print("Test 9 Passed.")

    # -------------------------------------------------------------
    # TEST 4: Insufficient Evidence Verification Gate
    # -------------------------------------------------------------
    print("\n--- Test 4: Insufficient Evidence Gate ---")
    res_insuf = reasoning_agent.reason(
        "What is the parental leave duration?",
        {
            "status": "insufficient",
            "is_sufficient": False,
            "reason": "Parental leave is not mentioned in the evidence.",
            "supporting_evidence": [],
        },
        sample_evidence,
    )
    print(json.dumps(res_insuf, indent=2))
    assert res_insuf["status"] == "insufficient"
    assert res_insuf["conclusion"] is None
    assert res_insuf["evidence_used"] == []
    print("Test 4 Passed.")

    # -------------------------------------------------------------
    # TEST 5: Conflicting Evidence Verification Gate
    # -------------------------------------------------------------
    print("\n--- Test 5: Conflicting Evidence Gate ---")
    res_conflict = reasoning_agent.reason(
        "How many days of annual leave do employees receive?",
        {
            "status": "conflicting",
            "is_sufficient": False,
            "reason": "The supplied evidence contains conflicting information.",
            "supporting_evidence": [],
        },
        sample_evidence,
    )
    print(json.dumps(res_conflict, indent=2))
    assert res_conflict["status"] == "conflicting"
    assert res_conflict["conclusion"] is None
    assert res_conflict["evidence_used"] == []
    print("Test 5 Passed.")

    # -------------------------------------------------------------
    # TEST 11: Verified Evidence Mismatch
    # -------------------------------------------------------------
    print("\n--- Test 11: Verified Evidence Mismatch ---")
    # Verification references chunk that is not in evidence list
    res_mismatch = reasoning_agent.reason(
        "How many days of annual leave do employees receive?",
        {
            "status": "sufficient",
            "is_sufficient": True,
            "supporting_evidence": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
        },
        [{"chunk_id": "page-3-chunk-1", "page_number": 3, "text": "Remote work policy."}],
    )
    print(json.dumps(res_mismatch, indent=2))
    assert res_mismatch["status"] == "insufficient"
    assert res_mismatch["conclusion"] is None
    assert "not found in supplied evidence" in res_mismatch["reason"]
    print("Test 11 Passed.")

    # -------------------------------------------------------------
    # TEST 8: Malformed LLM Output
    # -------------------------------------------------------------
    print("\n--- Test 8: Malformed LLM Output Handling ---")
    # Non-JSON
    try:
        reasoning_agent._parse_and_validate("Not JSON", "Query", sample_evidence)
        assert False, "Should raise ValueError on non-JSON"
    except ValueError as e:
        print(f"Non-JSON safely rejected: {e}")

    # Missing fields
    try:
        reasoning_agent._parse_and_validate(json.dumps({"status": "grounded"}), "Query", sample_evidence)
        assert False, "Should raise ValueError on missing fields"
    except ValueError as e:
        print(f"Missing fields safely rejected: {e}")

    # Fail safe through reason()
    original_call = reasoning_agent._call_ollama
    reasoning_agent._call_ollama = lambda prompt: "Bad response"
    res_fail_safe = reasoning_agent.reason(
        "Query",
        {"status": "sufficient", "is_sufficient": True, "supporting_evidence": [{"chunk_id": "page-2-chunk-1", "page_number": 2}]},
        sample_evidence,
    )
    assert res_fail_safe["status"] == "insufficient"
    assert res_fail_safe["conclusion"] is None
    assert "Reasoning failed due to parsing/validation error" in res_fail_safe["reason"]
    print("Fail-safe returned safely:", res_fail_safe)
    reasoning_agent._call_ollama = original_call

    # -------------------------------------------------------------
    # TEST 7: Unknown / Hallucinated Evidence Reference
    # -------------------------------------------------------------
    print("\n--- Test 7: Unknown Evidence Reference ---")
    mock_hallucinated_chunk = json.dumps({
        "status": "grounded",
        "conclusion": "Employees receive 24 days.",
        "reason": "Based on page 99.",
        "evidence_used": [{"chunk_id": "page-99-chunk-1", "page_number": 99}]
    })
    try:
        reasoning_agent._parse_and_validate(mock_hallucinated_chunk, "Query", sample_evidence)
        assert False, "Should reject unknown chunk ID"
    except ValueError as e:
        print(f"Unknown chunk ID safely rejected: {e}")

    print("\nAll unit tests passed successfully!")


def run_llm_reasoning_tests(
    verification_agent: EvidenceVerificationAgent,
    reasoning_agent: ReasoningAgent,
    chunks: list[dict],
):
    print("\n==================================================")
    print("RUNNING LLM REASONING TESTS")
    print("==================================================")

    # -------------------------------------------------------------
    # TEST 1: Annual Leave
    # -------------------------------------------------------------
    print("\n--- Test 1: Annual Leave ---")
    q1 = "How many days of annual leave do employees receive?"
    ev1 = [c for c in chunks if c.get("page_number") == 2 and "annual leave" in c.get("text", "").lower()]
    ver1 = verification_agent.verify(q1, ev1)
    assert ver1["is_sufficient"] is True

    res1 = reasoning_agent.reason(q1, ver1, ev1)
    print(json.dumps(res1, indent=2))

    assert res1["status"] == "grounded"
    assert res1["conclusion"] is not None
    assert "24" in res1["conclusion"]
    assert len(res1["evidence_used"]) > 0
    assert any(item["chunk_id"] == "page-2-chunk-1" for item in res1["evidence_used"])
    print("Test 1 Passed.")

    # -------------------------------------------------------------
    # TEST 2: Remote Work
    # -------------------------------------------------------------
    print("\n--- Test 2: Remote Work ---")
    q2 = "Can employees work remotely three days per week?"
    ev2 = [c for c in chunks if c.get("page_number") == 3 and "remote" in c.get("text", "").lower()]
    ver2 = verification_agent.verify(q2, ev2)
    assert ver2["is_sufficient"] is True

    res2 = reasoning_agent.reason(q2, ver2, ev2)
    print(json.dumps(res2, indent=2))

    assert res2["status"] == "grounded"
    assert res2["conclusion"] is not None
    assert "3 days" in res2["conclusion"].lower() or "three days" in res2["conclusion"].lower()
    assert "approval" in res2["conclusion"].lower()
    assert any(item["chunk_id"] == "page-3-chunk-1" for item in res2["evidence_used"])
    print("Test 2 Passed.")

    # -------------------------------------------------------------
    # TEST 3: Password Policy
    # -------------------------------------------------------------
    print("\n--- Test 3: Password Policy ---")
    q3 = "What is the password policy?"
    ev3 = [c for c in chunks if c.get("page_number") == 5 and "password" in c.get("text", "").lower()]
    ver3 = verification_agent.verify(q3, ev3)
    assert ver3["is_sufficient"] is True

    res3 = reasoning_agent.reason(q3, ver3, ev3)
    print(json.dumps(res3, indent=2))

    assert res3["status"] == "grounded"
    assert res3["conclusion"] is not None
    assert any(item["chunk_id"] == "page-5-chunk-1" for item in res3["evidence_used"])
    print("Test 3 Passed.")

    # -------------------------------------------------------------
    # TEST 6: Hallucination Resistance
    # -------------------------------------------------------------
    print("\n--- Test 6: Hallucination Resistance ---")
    # Supply minimal controlled chunk
    minimal_evidence = [
        {
            "chunk_id": "page-2-chunk-1",
            "page_number": 2,
            "text": "Employees receive 24 days of annual leave per calendar year."
        }
    ]
    ver6 = {
        "status": "sufficient",
        "is_sufficient": True,
        "supporting_evidence": [{"chunk_id": "page-2-chunk-1", "page_number": 2}],
    }
    res6 = reasoning_agent.reason(q1, ver6, minimal_evidence)
    print(json.dumps(res6, indent=2))

    assert res6["status"] == "grounded"
    conc = res6["conclusion"].lower()
    assert "24" in conc
    # Check that unsupported rules (probation, termination, unlimited carry-over) were not hallucinated
    assert "probation" not in conc, "Model hallucinated probation policy"
    assert "termination" not in conc, "Model hallucinated termination policy"
    print("Test 6 Passed.")


def main():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    print(f"Loaded {len(pages)} pages and {len(chunks)} chunks.")

    verification_agent = EvidenceVerificationAgent()
    reasoning_agent = ReasoningAgent()

    run_unit_tests(reasoning_agent)
    run_llm_reasoning_tests(verification_agent, reasoning_agent, chunks)

    print("\n==================================================")
    print("ALL REASONING AGENT TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
