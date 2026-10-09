import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.agents.evidence_verification_agent import EvidenceVerificationAgent


def run_unit_tests(agent: EvidenceVerificationAgent):
    print("==================================================")
    print("RUNNING UNIT TESTS (EDGE CASES & VALIDATION)")
    print("==================================================")

    # -------------------------------------------------------------
    # 1. Invalid Query
    # -------------------------------------------------------------
    print("\n--- Test 8: Invalid Query ---")
    try:
        agent.verify("", [{"chunk_id": "c1", "page_number": 1, "text": "sample"}])
        assert False, "Should raise ValueError on empty query"
    except ValueError as e:
        print(f"Empty query caught as expected: {e}")

    try:
        agent.verify(None, [{"chunk_id": "c1", "page_number": 1, "text": "sample"}])
        assert False, "Should raise ValueError on None query"
    except ValueError as e:
        print(f"None query caught as expected: {e}")

    # -------------------------------------------------------------
    # 2. Empty Evidence
    # -------------------------------------------------------------
    print("\n--- Test 6: Empty Evidence ---")
    res_empty = agent.verify("What is the parental leave duration?", [])
    print(json.dumps(res_empty, indent=2))
    assert res_empty["is_sufficient"] is False
    assert res_empty["status"] == "insufficient"
    assert res_empty["confidence"] == 1.0
    assert "No retrieved evidence was provided" in res_empty["reason"]
    assert res_empty["supporting_evidence"] == []
    print("Empty evidence test passed.")

    # -------------------------------------------------------------
    # 3. Malformed LLM Output Handling
    # -------------------------------------------------------------
    print("\n--- Test 9: Malformed LLM Output Handling ---")
    sample_evidence = [{"chunk_id": "c1", "page_number": 1, "text": "Some text."}]

    # Non-JSON
    res_malformed = agent._parse_and_validate("Not JSON at all", "What is policy?", sample_evidence) \
        if False else None
    try:
        agent._parse_and_validate("Not JSON at all", "What is policy?", sample_evidence)
        assert False, "Should raise ValueError on non-JSON"
    except ValueError as e:
        print(f"Non-JSON output safely rejected: {e}")

    # Missing fields
    try:
        agent._parse_and_validate(json.dumps({"is_sufficient": True}), "What is policy?", sample_evidence)
        assert False, "Should raise ValueError on missing fields"
    except ValueError as e:
        print(f"Missing fields safely rejected: {e}")

    # Safe failure test via verify() fallback
    # Monkey-patch _call_ollama to return gibberish
    original_call = agent._call_ollama
    agent._call_ollama = lambda prompt: "Gibberish <bad> json"
    res_fail_safe = agent.verify("What is policy?", sample_evidence)
    assert res_fail_safe["is_sufficient"] is False
    assert res_fail_safe["confidence"] == 0.0
    assert "failed due to parsing/validation error" in res_fail_safe["reason"]
    print("Verify() fail-safe returned:", res_fail_safe)
    agent._call_ollama = original_call

    # -------------------------------------------------------------
    # 4. Invalid Confidence Validation
    # -------------------------------------------------------------
    print("\n--- Test 10: Invalid Confidence (> 1.0 or < 0.0) ---")
    bad_conf_high = json.dumps({
        "is_sufficient": True,
        "confidence": 1.5,
        "reason": "Test reason",
        "supporting_evidence": [{"chunk_id": "c1", "page_number": 1}]
    })
    try:
        agent._parse_and_validate(bad_conf_high, "query", sample_evidence)
        assert False, "Should reject confidence > 1.0"
    except ValueError as e:
        print(f"Confidence > 1.0 safely rejected: {e}")

    bad_conf_low = json.dumps({
        "is_sufficient": True,
        "confidence": -0.2,
        "reason": "Test reason",
        "supporting_evidence": [{"chunk_id": "c1", "page_number": 1}]
    })
    try:
        agent._parse_and_validate(bad_conf_low, "query", sample_evidence)
        assert False, "Should reject confidence < 0.0"
    except ValueError as e:
        print(f"Confidence < 0.0 safely rejected: {e}")

    # -------------------------------------------------------------
    # 5. Hallucinated / Unknown Supporting Chunk ID
    # -------------------------------------------------------------
    print("\n--- Test 11: Unknown Supporting Chunk ID ---")
    bad_chunk_json = json.dumps({
        "is_sufficient": True,
        "confidence": 0.9,
        "reason": "Test reason",
        "supporting_evidence": [{"chunk_id": "page-7-chunk-1", "page_number": 7}]
    })
    try:
        agent._parse_and_validate(bad_chunk_json, "query", sample_evidence)
        assert False, "Should reject unknown chunk ID"
    except ValueError as e:
        print(f"Hallucinated chunk ID safely rejected: {e}")

    print("\nAll unit tests passed successfully!")


def run_llm_verification_tests(agent: EvidenceVerificationAgent, chunks: list[dict]):
    print("\n==================================================")
    print("RUNNING LLM VERIFICATION TESTS")
    print("==================================================")

    chunk_map = {c["chunk_id"]: c for c in chunks}

    # -------------------------------------------------------------
    # 1. Directly Supported Annual Leave Question
    # -------------------------------------------------------------
    print("\n--- Test 1: Annual Leave (Direct Support) ---")
    q1 = "How many days of annual leave do employees receive?"
    ev1 = [c for c in chunks if c.get("page_number") == 2 and "annual leave" in c.get("text", "").lower()]
    assert len(ev1) > 0, "Page 2 annual leave chunk not found"
    res1 = agent.verify(q1, ev1)
    print(json.dumps(res1, indent=2))

    assert res1["is_sufficient"] is True
    assert res1["status"] == "sufficient"
    assert 0.0 <= res1["confidence"] <= 1.0
    assert len(res1["supporting_evidence"]) > 0
    assert any(s["page_number"] == 2 for s in res1["supporting_evidence"])
    print("Test 1 Passed.")

    # -------------------------------------------------------------
    # 2. Parental Leave Insufficient Evidence
    # -------------------------------------------------------------
    print("\n--- Test 2: Parental Leave Insufficient Evidence ---")
    q2 = "What is the parental leave duration?"
    # Pass only annual leave chunk
    res2 = agent.verify(q2, ev1)
    print(json.dumps(res2, indent=2))

    assert res2["is_sufficient"] is False
    assert res2["status"] == "insufficient"
    assert 0.0 <= res2["confidence"] <= 1.0
    assert res2["supporting_evidence"] == []
    print("Test 2 Passed.")

    # -------------------------------------------------------------
    # 3. International Travel Insufficient Evidence
    # -------------------------------------------------------------
    print("\n--- Test 3: International Travel Reimbursement Limit ---")
    q3 = "What is the international travel reimbursement limit?"
    # Provide domestic travel / general expense chunk
    ev3 = [
        {
            "chunk_id": "page-4-chunk-1",
            "page_number": 4,
            "text": "Economy-class air travel is permitted for domestic business trips. Hotel expenses up to $200 per night are reimbursable for domestic trips."
        }
    ]
    res3 = agent.verify(q3, ev3)
    print(json.dumps(res3, indent=2))

    assert res3["is_sufficient"] is False
    assert res3["status"] == "insufficient"
    assert 0.0 <= res3["confidence"] <= 1.0
    assert res3["supporting_evidence"] == []
    print("Test 3 Passed.")

    # -------------------------------------------------------------
    # 4. Remote Work Sufficient Evidence
    # -------------------------------------------------------------
    print("\n--- Test 4: Remote Work (Direct Support) ---")
    q4 = "Can employees work remotely three days per week?"
    ev4 = [c for c in chunks if c.get("page_number") == 3 and "remote" in c.get("text", "").lower()]
    assert len(ev4) > 0, "Page 3 remote work chunk not found"
    res4 = agent.verify(q4, ev4)
    print(json.dumps(res4, indent=2))

    assert res4["is_sufficient"] is True
    assert res4["status"] == "sufficient"
    assert 0.0 <= res4["confidence"] <= 1.0
    assert len(res4["supporting_evidence"]) > 0
    assert any(s["page_number"] == 3 for s in res4["supporting_evidence"])
    print("Test 4 Passed.")

    # -------------------------------------------------------------
    # 5. Password Policy Sufficient Evidence
    # -------------------------------------------------------------
    print("\n--- Test 5: Password Policy ---")
    q5 = "What is the password policy?"
    ev5 = [c for c in chunks if c.get("page_number") == 5 and "password" in c.get("text", "").lower()]
    assert len(ev5) > 0, "Page 5 password policy chunk not found"
    res5 = agent.verify(q5, ev5)
    print(json.dumps(res5, indent=2))

    assert res5["is_sufficient"] is True
    assert res5["status"] == "sufficient"
    assert 0.0 <= res5["confidence"] <= 1.0
    assert len(res5["supporting_evidence"]) > 0
    assert any(s["page_number"] == 5 for s in res5["supporting_evidence"])
    print("Test 5 Passed.")

    # -------------------------------------------------------------
    # 7. Conflicting Evidence Test
    # -------------------------------------------------------------
    print("\n--- Test 7: Conflicting Evidence ---")
    q7 = "How many days of annual leave do employees receive?"
    conflicting_evidence = [
        {
            "chunk_id": "policy-chunk-a",
            "page_number": 2,
            "text": "Full-time employees receive 24 days of annual leave per calendar year."
        },
        {
            "chunk_id": "policy-chunk-b",
            "page_number": 2,
            "text": "Full-time employees receive 20 days of annual leave per calendar year."
        }
    ]
    res7 = agent.verify(q7, conflicting_evidence)
    print(json.dumps(res7, indent=2))

    # Must be either is_sufficient = False or status = 'conflicting'
    assert res7["is_sufficient"] is False
    assert res7["status"] in ("conflicting", "insufficient")
    assert res7["supporting_evidence"] == []
    print("Test 7 Passed.")


def main():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")

    agent = EvidenceVerificationAgent()

    run_unit_tests(agent)
    run_llm_verification_tests(agent, chunks)

    print("\n==================================================")
    print("ALL EVIDENCE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()
