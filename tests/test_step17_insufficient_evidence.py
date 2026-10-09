import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.agents.evidence_verification_agent import EvidenceVerificationAgent


def run_step_17_tests(agent: EvidenceVerificationAgent, chunks: list[dict]):
    print("==================================================")
    print("RUNNING STEP 17 ADVANCED INSUFFICIENT-EVIDENCE TESTS")
    print("==================================================")

    # -------------------------------------------------------------
    # TEST 1: How many annual leave days are provided? (Sufficient)
    # -------------------------------------------------------------
    print("\n--- Test 1: Annual Leave Days Provided ---")
    q1 = "How many annual leave days are provided?"
    ev1 = [c for c in chunks if c.get("page_number") == 2 and "annual leave" in c.get("text", "").lower()]
    assert len(ev1) > 0, "Page 2 annual leave chunk not found"
    res1 = agent.verify(q1, ev1)
    print(json.dumps(res1, indent=2))

    assert res1["is_sufficient"] is True
    assert res1["status"] == "sufficient"
    assert len(res1["supporting_evidence"]) > 0
    assert any(s["page_number"] == 2 for s in res1["supporting_evidence"])
    print("Test 1 Passed.")

    # -------------------------------------------------------------
    # TEST 2: How long is parental leave? (Insufficient)
    # -------------------------------------------------------------
    print("\n--- Test 2: Parental Leave Duration ---")
    q2 = "How long is parental leave?"
    # Supply annual leave & sick leave chunks from page 2 (leave related but not parental leave)
    res2 = agent.verify(q2, ev1)
    print(json.dumps(res2, indent=2))

    assert res2["is_sufficient"] is False
    assert res2["status"] == "insufficient"
    assert res2["supporting_evidence"] == []
    print("Test 2 Passed.")

    # -------------------------------------------------------------
    # TEST 3: Partial-Answer / Compound Question (Insufficient)
    # "What are the remote-work eligibility requirements and how many days per week are allowed?"
    # Evidence supports days (3 days) but not eligibility criteria/requirements.
    # -------------------------------------------------------------
    print("\n--- Test 3: Compound Question (Partial Answer) ---")
    q3 = "What are the remote-work eligibility requirements and how many days per week are allowed?"
    ev3 = [c for c in chunks if c.get("page_number") == 3 and "remote" in c.get("text", "").lower()]
    assert len(ev3) > 0, "Page 3 remote work chunk not found"
    res3 = agent.verify(q3, ev3)
    print(json.dumps(res3, indent=2))

    assert res3["is_sufficient"] is False
    assert res3["status"] == "insufficient"
    assert res3["supporting_evidence"] == []
    print("Test 3 Passed.")

    # -------------------------------------------------------------
    # TEST 4: International Travel Reimbursement Limit (Specific Value Missing)
    # -------------------------------------------------------------
    print("\n--- Test 4: International Travel Reimbursement Limit ---")
    q4 = "What is the international travel reimbursement limit?"
    ev4 = [c for c in chunks if c.get("page_number") == 4 and "travel" in c.get("text", "").lower()]
    assert len(ev4) > 0, "Page 4 travel chunk not found"
    res4 = agent.verify(q4, ev4)
    print(json.dumps(res4, indent=2))

    assert res4["is_sufficient"] is False
    assert res4["status"] == "insufficient"
    assert res4["supporting_evidence"] == []
    print("Test 4 Passed.")

    # -------------------------------------------------------------
    # TEST 5: Password Policy (Sufficient)
    # -------------------------------------------------------------
    print("\n--- Test 5: Password Policy ---")
    q5 = "What is the password policy?"
    ev5 = [c for c in chunks if c.get("page_number") == 5 and "password" in c.get("text", "").lower()]
    assert len(ev5) > 0, "Page 5 password chunk not found"
    res5 = agent.verify(q5, ev5)
    print(json.dumps(res5, indent=2))

    assert res5["is_sufficient"] is True
    assert res5["status"] == "sufficient"
    assert len(res5["supporting_evidence"]) > 0
    assert any(s["page_number"] == 5 for s in res5["supporting_evidence"])
    print("Test 5 Passed.")

    # -------------------------------------------------------------
    # TEST 6: Relevant-Looking but Non-Supporting Evidence (Keyword False Positive)
    # -------------------------------------------------------------
    print("\n--- Test 6: Relevant-Looking but Non-Supporting Evidence ---")
    q6 = "What is the performance-review schedule?"
    # Supply chunk with enterprise words like employee, review, etc.
    ev6 = [
        {
            "chunk_id": "page-4-chunk-2",
            "page_number": 4,
            "text": "Routine business expenses must be submitted through the expense management system. Managers are responsible for reviewing expense claims before Finance processes reimbursement."
        }
    ]
    res6 = agent.verify(q6, ev6)
    print(json.dumps(res6, indent=2))

    assert res6["is_sufficient"] is False
    assert res6["status"] == "insufficient"
    assert res6["supporting_evidence"] == []
    print("Test 6 Passed.")

    # -------------------------------------------------------------
    # TEST 7: Conflicting Evidence (Conflicting)
    # -------------------------------------------------------------
    print("\n--- Test 7: Conflicting Evidence ---")
    q7 = "How many annual leave days are provided?"
    conflicting_ev = [
        {
            "chunk_id": "c-leave-1",
            "page_number": 2,
            "text": "Employees receive 24 days of annual leave per calendar year."
        },
        {
            "chunk_id": "c-leave-2",
            "page_number": 2,
            "text": "Employees receive 30 days of annual leave per calendar year."
        }
    ]
    res7 = agent.verify(q7, conflicting_ev)
    print(json.dumps(res7, indent=2))

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
    print(f"Loaded {len(pages)} pages and {len(chunks)} chunks.")

    agent = EvidenceVerificationAgent()
    run_step_17_tests(agent, chunks)

    print("\n==================================================")
    print("ALL STEP 17 INSUFFICIENT-EVIDENCE TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
