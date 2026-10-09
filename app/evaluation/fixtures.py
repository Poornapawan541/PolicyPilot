"""
Step 21 Evaluation Fixtures Loader.

Provides synthetic conflict and supersession chunks for evaluation items EVAL-019 through EVAL-022
that test cross-document contradictions and supersession, exactly matching the verified test fixtures
from Step 18 and Step 19.
"""

from typing import Optional


def get_evaluation_fixture_chunks(eval_id: str) -> list[dict]:
    """
    Returns specific synthetic fixture chunks for evaluation records that test
    cross-document conflicts or explicit supersession without polluting the base document corpus.
    """
    if eval_id == "EVAL-019":
        # Numerical conflict: Remote work 3 days (Handbook) vs 2 days (Addendum conflict chunk)
        return [
            {
                "chunk_id": "PP-TEST-002-page-1-conflict",
                "document_id": "PP-TEST-002",
                "document_name": "PolicyPilot_Test_Remote_Work_Addendum.pdf",
                "page_number": 1,
                "text": "Eligible employees may work remotely up to 2 days per week with direct manager approval.",
            }
        ]

    elif eval_id == "EVAL-020":
        # Monetary conflict: Domestic hotel reimbursement limit (Handbook INR 4,500 vs Update INR 6,000)
        return [
            {
                "chunk_id": "PP-TEST-003-page-2-chunk-1",
                "document_id": "PP-TEST-003",
                "document_name": "PolicyPilot_Test_Travel_Policy_Update.pdf",
                "page_number": 2,
                "text": "Domestic hotel reimbursement is limited to INR 6,000 per night for all employees.",
            }
        ]

    elif eval_id == "EVAL-021":
        # Leave conflict: Annual leave entitlement (Handbook 24 days vs Amendment 20 days)
        return [
            {
                "chunk_id": "PP-TEST-005-page-1-chunk-1",
                "document_id": "PP-TEST-005",
                "document_name": "PolicyPilot_Test_Leave_Amendment.pdf",
                "page_number": 1,
                "text": "Full-time employees receive 20 days of paid annual leave per calendar year.",
            }
        ]

    elif eval_id == "EVAL-022":
        # Explicit supersession: Remote work up to 4 days with explicit supersession clause
        return [
            {
                "chunk_id": "PP-TEST-004-page-1-chunk-1",
                "document_id": "PP-TEST-004",
                "document_name": "PolicyPilot_Test_Remote_Policy_2026.pdf",
                "page_number": 1,
                "text": "This document supersedes all prior remote work policies. Employees may work remotely up to 4 days per week.",
            }
        ]

    return []
