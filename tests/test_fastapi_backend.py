"""
Deterministic unit and integration tests for PolicyPilot Step 23 FastAPI Backend.

Tests verify:
1. /health endpoint: Instant 200 OK without loading any ML models or LangGraph.
2. /ask endpoint:
   - Grounded prediction with citations and trace information.
   - Insufficient evidence prediction.
   - Conflicting evidence prediction.
   - Trace disabled option (enable_tracing=False).
   - Empty/whitespace question validation (HTTP 400).
   - Malformed JSON request body validation (HTTP 422).
   - Internal workflow error handling without leaking stack traces (HTTP 500).
   - Model reuse / singleton verification (pipeline_manager initialized once).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.api.pipeline import pipeline_manager, PipelineManager


client = TestClient(app)


def test_health_endpoint_no_models_loaded():
    print("\n--- Test 1: Health Endpoint (No Heavy Models Loaded) ---")
    # Reset pipeline manager state to ensure health check does not initialize models
    pipeline_manager._graph = None
    pipeline_manager._initialized = False

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "PolicyPilot"
    assert data["version"] == "1.0.0"

    # Confirm that heavy models / graph were NOT loaded
    assert pipeline_manager.is_initialized() is False
    print("Passed Test 1: /health returned 200 OK instantly without loading models.")


def test_ask_grounded_response():
    print("\n--- Test 2: Ask Endpoint with Grounded Response ---")
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "status": "grounded",
        "final_answer": "Employees receive 24 days of annual leave per calendar year. [Page 2]",
        "response_result": {
            "status": "grounded",
            "answer": "Employees receive 24 days of annual leave per calendar year. [Page 2]",
            "citations": [
                {
                    "citation": "[Page 2]",
                    "chunk_id": "c1",
                    "document_id": "PP-TEST-001",
                    "document_name": "PolicyPilot_Test_Enterprise_Handbook.pdf",
                    "page_number": 2,
                }
            ],
        },
        "trace_id": "trace-test-123",
        "trace": {
            "trace_id": "trace-test-123",
            "enabled": True,
            "final_status": "grounded",
            "total_duration_ms": 120.5,
            "stages": [{"stage": "verification", "duration_ms": 50.0}],
            "_tracer_obj": None,  # Should be filtered out in API response
        },
    }

    pipeline_manager.set_graph(mock_graph)

    req_payload = {
        "question": "How many days of annual leave do employees receive?",
        "trace_id": "trace-test-123",
        "enable_tracing": True,
    }
    response = client.post("/ask", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["question"] == req_payload["question"]
    assert data["status"] == "grounded"
    assert "24 days" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation"] == "[Page 2]"
    assert data["citations"][0]["document_id"] == "PP-TEST-001"
    assert data["citations"][0]["page_number"] == 2
    assert data["trace_id"] == "trace-test-123"
    assert data["trace"]["enabled"] is True
    assert "_tracer_obj" not in data["trace"]

    print("Passed Test 2: /ask returned structured grounded response with citations.")


def test_ask_insufficient_evidence():
    print("\n--- Test 3: Ask Endpoint with Insufficient Evidence ---")
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "status": "insufficient",
        "final_answer": "Insufficient evidence in the provided documents to answer this question.",
        "response_result": {
            "status": "insufficient",
            "answer": "Insufficient evidence in the provided documents to answer this question.",
            "citations": [],
        },
        "trace_id": "trace-insuf-456",
        "trace": {
            "trace_id": "trace-insuf-456",
            "enabled": True,
            "final_status": "insufficient",
            "total_duration_ms": 85.0,
        },
    }
    pipeline_manager.set_graph(mock_graph)

    response = client.post("/ask", json={"question": "What is the parental leave duration?"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "insufficient"
    assert "Insufficient evidence" in data["answer"]
    assert data["citations"] == []
    print("Passed Test 3: /ask returned insufficient status without hallucinated citations.")


def test_ask_conflicting_evidence():
    print("\n--- Test 4: Ask Endpoint with Conflicting Evidence ---")
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "status": "conflicting",
        "final_answer": "Policy conflict detected between retrieved sources. Unreconciled contradictory claims exist.",
        "response_result": {
            "status": "conflicting",
            "answer": "Policy conflict detected between retrieved sources. Unreconciled contradictory claims exist.",
            "citations": [],
        },
        "trace_id": "trace-conf-789",
        "trace": {
            "trace_id": "trace-conf-789",
            "enabled": True,
            "final_status": "conflicting",
            "total_duration_ms": 92.0,
        },
    }
    pipeline_manager.set_graph(mock_graph)

    response = client.post("/ask", json={"question": "What is the conflicting remote work allowance?"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "conflicting"
    assert "conflict detected" in data["answer"].lower()
    print("Passed Test 4: /ask returned conflicting status safely.")


def test_ask_disabled_tracing():
    print("\n--- Test 5: Ask Endpoint with Disabled Tracing ---")
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "status": "grounded",
        "final_answer": "Password must be at least 12 characters. [Page 5]",
        "response_result": {"status": "grounded", "answer": "Password must be at least 12 characters. [Page 5]"},
        "trace_id": "trace-999",
        "trace": {"trace_id": "trace-999", "enabled": False},
    }
    pipeline_manager.set_graph(mock_graph)

    response = client.post(
        "/ask",
        json={"question": "What is the password length requirement?", "enable_tracing": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trace"] is None
    print("Passed Test 5: /ask omitted trace payload when enable_tracing=False.")


def test_empty_or_whitespace_question_validation():
    print("\n--- Test 6: Empty and Malformed Questions ---")
    # Empty string
    r1 = client.post("/ask", json={"question": ""})
    assert r1.status_code == 422 or r1.status_code == 400

    # Whitespace only
    r2 = client.post("/ask", json={"question": "   \n\t  "})
    assert r2.status_code == 400
    assert "Question must not be empty" in r2.json()["detail"]

    # Missing question field altogether
    r3 = client.post("/ask", json={})
    assert r3.status_code == 422

    print("Passed Test 6: Empty and malformed questions rejected with 400/422 status.")


def test_internal_workflow_error_handling():
    print("\n--- Test 7: Internal Workflow Error Handling ---")
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("Simulated internal workflow failure")
    pipeline_manager.set_graph(mock_graph)

    response = client.post("/ask", json={"question": "Will this crash safely?"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Internal server error" in detail
    assert "Simulated internal workflow failure" in detail
    print("Passed Test 7: Workflow exceptions caught safely with HTTP 500.")


def test_model_reuse_across_requests():
    print("\n--- Test 8: Model and Graph Singleton Reuse ---")
    manager = PipelineManager()
    assert manager.is_initialized() is False

    mock_graph_singleton = MagicMock()
    manager.set_graph(mock_graph_singleton)
    assert manager.is_initialized() is True

    # Repeated calls should return the same instance
    g1 = manager.get_graph()
    g2 = manager.get_graph()
    assert g1 is g2
    assert g1 is mock_graph_singleton

    print("Passed Test 8: PipelineManager reuses compiled graph without reloading.")


def test_cors_production_origin_allowed():
    print("\n--- Test 9: CORS Production and Localhost Allowed ---")
    headers = {
        "Origin": "https://policy-pilot-eta.vercel.app",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://policy-pilot-eta.vercel.app"

    # Localhost origin check
    lh_headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    lh_res = client.options("/health", headers=lh_headers)
    assert lh_res.status_code == 200
    assert lh_res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    print("Passed Test 9: Production origin and localhost correctly allowed.")


def test_cors_preview_origin_allowed():
    print("\n--- Test 10: CORS Tightly Constrained Preview Allowed ---")
    preview_headers = {
        "Origin": "https://policy-pilot-fullstack-5x4y9bbnm-skill-swap12.vercel.app",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/health", headers=preview_headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://policy-pilot-fullstack-5x4y9bbnm-skill-swap12.vercel.app"
    print("Passed Test 10: PolicyPilot project preview origin allowed.")


def test_cors_arbitrary_origin_rejected():
    print("\n--- Test 11: CORS Arbitrary Origin Rejected ---")
    bad_headers = {
        "Origin": "https://evil-unrelated-site.com",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/health", headers=bad_headers)
    # When CORS rejects an origin, FastAPI does not echo access-control-allow-origin
    assert "access-control-allow-origin" not in response.headers

    # Also test an unrelated vercel app that doesn't match the prefix policy
    unrelated_vercel = {
        "Origin": "https://random-unrelated-app.vercel.app",
        "Access-Control-Request-Method": "POST",
    }
    uv_res = client.options("/health", headers=unrelated_vercel)
    assert "access-control-allow-origin" not in uv_res.headers
    print("Passed Test 11: Unrelated origins cleanly rejected by CORS.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING POLICYPILOT STEP 23 FASTAPI BACKEND TESTS")
    print("=" * 60)
    test_health_endpoint_no_models_loaded()
    test_ask_grounded_response()
    test_ask_insufficient_evidence()
    test_ask_conflicting_evidence()
    test_ask_disabled_tracing()
    test_empty_or_whitespace_question_validation()
    test_internal_workflow_error_handling()
    test_model_reuse_across_requests()
    test_cors_production_origin_allowed()
    test_cors_preview_origin_allowed()
    test_cors_arbitrary_origin_rejected()
    print("=" * 60)
    print("ALL STEP 23 FASTAPI BACKEND TESTS PASSED!")
    print("=" * 60)
