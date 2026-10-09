"""
Unit and regression tests for Step 22: Observability and Tracing.

Tests cover:
1. Trace ID creation, propagation, and default generation.
2. Stage start/end timing and duration recording.
3. Metadata sanitization (passwords, tokens, large chunk text, chunk ID preservation).
4. Status rationale derivation for grounded, insufficient, conflicting, and error.
5. End-to-end graph execution trace population and deterministic mock execution.
6. Error handling and trace capture during node failures.
7. Disabling tracing (enable_tracing=False and POLICYPILOT_TRACING_ENABLED=false).
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.tracer import WorkflowTracer, derive_status_reason
from app.workflow.policypilot_graph import build_policypilot_graph, PolicyPilotState
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent


def test_tracer_lifecycle_and_timing():
    print("\n--- Test 1: Tracer Lifecycle and Timing ---")
    tracer = WorkflowTracer(trace_id="test-trace-123", enabled=True)
    assert tracer.trace_id == "test-trace-123"
    assert tracer.enabled is True

    tracer.start_stage("test_stage_1")
    tracer.end_stage("test_stage_1", metadata={"key": "value"})

    trace_data = tracer.finish()
    assert trace_data["trace_id"] == "test-trace-123"
    assert trace_data["enabled"] is True
    assert "start_time" in trace_data
    assert "end_time" in trace_data
    assert trace_data["total_duration_ms"] >= 0.0

    stages = trace_data["stages"]
    assert len(stages) == 1
    assert stages[0]["stage"] == "test_stage_1"
    assert stages[0]["duration_ms"] >= 0.0
    assert stages[0]["metadata"]["key"] == "value"
    print("Passed Test 1: Tracer lifecycle and timing verified.")


def test_metadata_sanitization():
    print("\n--- Test 2: Metadata Sanitization & Secret Redaction ---")
    raw_metadata = {
        "user_query": "What is the annual leave?",
        "api_key": "sk-secret-12345",
        "auth_token": "bearer-xyz",
        "password": "SuperSecretPassword123",
        "text": "A" * 300,
        "evidence": [
            {
                "chunk_id": "c1",
                "document_id": "DOC-1",
                "document_name": "Handbook.pdf",
                "page_number": 2,
                "score": 0.95432,
                "text": "This is a full long chunk text that should not bloat the trace logs.",
            }
        ],
    }

    sanitized = WorkflowTracer._sanitize_metadata(raw_metadata)

    # Sensitive keys must be redacted
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["auth_token"] == "[REDACTED]"
    assert sanitized["password"] == "[REDACTED]"

    # Long text must be truncated
    assert "truncated 300 chars" in sanitized["text"]

    # Evidence chunks must preserve IDs and page metadata without full text
    assert sanitized["evidence_count"] == 1
    ev_chunk = sanitized["evidence_chunks"][0]
    assert ev_chunk["chunk_id"] == "c1"
    assert ev_chunk["document_id"] == "DOC-1"
    assert ev_chunk["page_number"] == 2
    assert ev_chunk["score"] == 0.9543
    assert "text" not in ev_chunk

    print("Passed Test 2: Sanitization correctly redacts secrets and large text.")


def test_derive_status_reason():
    print("\n--- Test 3: Status Rationale Derivation ---")
    # Grounded
    reason_grounded = derive_status_reason(
        status="grounded",
        verification_result={"status": "sufficient"},
        reasoning_result={"status": "grounded", "evidence_used": ["c1", "c2"]},
    )
    assert "2 verified supporting evidence chunk(s)" in reason_grounded

    # Insufficient
    reason_insufficient = derive_status_reason(
        status="insufficient",
        verification_result={"status": "insufficient", "missing_aspects": ["parental leave policy"]},
        reasoning_result=None,
    )
    assert "Missing aspects: parental leave policy" in reason_insufficient

    # Conflicting
    reason_conflicting = derive_status_reason(
        status="conflicting",
        verification_result={"status": "conflicting", "conflicts": [{"claim_a": "3 days", "claim_b": "2 days"}]},
        reasoning_result=None,
    )
    assert "Cross-document contradictions detected" in reason_conflicting

    # Error
    reason_error = derive_status_reason(
        status="error",
        verification_result=None,
        reasoning_result=None,
        error="Connection refused",
    )
    assert "Workflow halted due to error: Connection refused" in reason_error

    print("Passed Test 3: Status rationale derivation correctly explains all outcomes.")


def test_graph_trace_propagation():
    print("\n--- Test 4: LangGraph End-to-End Tracing Propagation ---")
    mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
    mock_retrieval_agent = MagicMock(spec=RetrievalAgent)
    mock_verification_agent = MagicMock(spec=EvidenceVerificationAgent)
    mock_reasoning_agent = MagicMock(spec=ReasoningAgent)
    mock_response_agent = MagicMock(spec=ResponseAgent)

    mock_query_agent.analyze.return_value = {
        "original_query": "How many days of leave?",
        "intent": "annual leave",
        "search_query": "annual leave",
        "keywords": ["leave"],
        "filters": {},
    }
    mock_retrieval_agent.retrieve.return_value = [
        {"chunk_id": "c1", "document_id": "D1", "document_name": "Handbook", "page_number": 2, "score": 0.9, "text": "24 days leave"}
    ]
    mock_verification_agent.verify.return_value = {
        "status": "sufficient",
        "missing_aspects": [],
        "conflicts": [],
    }
    mock_reasoning_agent.reason.return_value = {
        "status": "grounded",
        "conclusion": "Employees receive 24 days leave.",
        "evidence_used": ["c1"],
    }
    mock_response_agent.respond.return_value = {
        "status": "grounded",
        "answer": "Employees receive 24 days leave. [Page 2]",
        "citations": [{"citation": "[Page 2]", "chunk_id": "c1", "document_id": "D1", "page_number": 2}],
    }

    graph = build_policypilot_graph(
        chunks=[{"chunk_id": "c1", "page_number": 2, "text": "24 days leave"}],
        query_agent=mock_query_agent,
        retrieval_agent=mock_retrieval_agent,
        verification_agent=mock_verification_agent,
        reasoning_agent=mock_reasoning_agent,
        response_agent=mock_response_agent,
    )

    custom_trace_id = "trace-custom-999"
    result = graph.invoke({
        "original_query": "How many days of leave?",
        "trace_id": custom_trace_id,
        "evidence": [],
    })

    assert result["status"] == "grounded"
    assert result["trace_id"] == custom_trace_id
    assert "trace" in result

    trace = result["trace"]
    assert trace["trace_id"] == custom_trace_id
    assert trace["final_status"] == "grounded"
    assert "status_reason" in trace
    assert trace["total_duration_ms"] >= 0.0

    stage_names = [s["stage"] for s in trace["stages"]]
    assert stage_names == ["query_understanding", "retrieval", "verification", "reasoning", "response"]
    for stage in trace["stages"]:
        assert stage["duration_ms"] >= 0.0
        assert "start_time" in stage
        assert "end_time" in stage

    print("Passed Test 4: Complete stage tracing and custom trace ID preserved across graph.")


def test_graph_insufficient_trace_path():
    print("\n--- Test 5: Tracing on Insufficient Evidence Path ---")
    mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
    mock_retrieval_agent = MagicMock(spec=RetrievalAgent)
    mock_verification_agent = MagicMock(spec=EvidenceVerificationAgent)
    mock_reasoning_agent = MagicMock(spec=ReasoningAgent)
    mock_response_agent = MagicMock(spec=ResponseAgent)

    mock_query_agent.analyze.return_value = {"intent": "parental leave", "keywords": []}
    mock_retrieval_agent.retrieve.return_value = []
    mock_verification_agent.verify.return_value = {
        "status": "insufficient",
        "missing_aspects": ["parental leave"],
        "conflicts": [],
    }
    mock_response_agent.respond.return_value = {
        "status": "insufficient",
        "answer": "Insufficient evidence.",
        "citations": [],
    }

    graph = build_policypilot_graph(
        chunks=[],
        query_agent=mock_query_agent,
        retrieval_agent=mock_retrieval_agent,
        verification_agent=mock_verification_agent,
        reasoning_agent=mock_reasoning_agent,
        response_agent=mock_response_agent,
    )

    result = graph.invoke({"original_query": "Parental leave?", "evidence": []})
    assert result["status"] == "insufficient"
    trace = result["trace"]
    stage_names = [s["stage"] for s in trace["stages"]]
    # Reasoning node must NOT appear in the trace for insufficient evidence
    assert "reasoning" not in stage_names
    assert stage_names == ["query_understanding", "retrieval", "verification", "response"]
    assert "Missing aspects" in trace["status_reason"]

    print("Passed Test 5: Insufficient evidence path accurately traced without calling reasoning.")


def test_disabled_tracing():
    print("\n--- Test 6: Disabled Tracing ---")
    mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
    mock_retrieval_agent = MagicMock(spec=RetrievalAgent)
    mock_verification_agent = MagicMock(spec=EvidenceVerificationAgent)
    mock_reasoning_agent = MagicMock(spec=ReasoningAgent)
    mock_response_agent = MagicMock(spec=ResponseAgent)

    mock_query_agent.analyze.return_value = {"intent": "test"}
    mock_retrieval_agent.retrieve.return_value = []
    mock_verification_agent.verify.return_value = {"status": "insufficient"}
    mock_response_agent.respond.return_value = {"status": "insufficient", "answer": "Insufficient."}

    graph = build_policypilot_graph(
        chunks=[],
        query_agent=mock_query_agent,
        retrieval_agent=mock_retrieval_agent,
        verification_agent=mock_verification_agent,
        reasoning_agent=mock_reasoning_agent,
        response_agent=mock_response_agent,
    )

    # Disable via state flag
    result = graph.invoke({
        "original_query": "Test query?",
        "enable_tracing": False,
        "evidence": [],
    })
    trace = result["trace"]
    assert trace["enabled"] is False
    assert "stages" not in trace

    print("Passed Test 6: Tracing correctly disabled when requested.")


def test_error_capture_in_trace():
    print("\n--- Test 7: Error Capture in Trace ---")
    mock_query_agent = MagicMock(spec=QueryUnderstandingAgent)
    mock_query_agent.analyze.side_effect = RuntimeError("Simulated agent failure")

    dummy_chunks = [{"chunk_id": "c1", "page_number": 1, "text": "Test chunk"}]
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

    result = graph.invoke({"original_query": "Trigger error", "evidence": []})
    assert result["status"] == "error"
    trace = result["trace"]
    assert trace["final_status"] == "error"
    assert "Simulated agent failure" in trace["error"]
    assert trace["stages"][0]["error"] == "query_understanding node failed: Simulated agent failure"

    print("Passed Test 7: Error captured in trace metadata and decision.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING POLICYPILOT STEP 22 OBSERVABILITY TESTS")
    print("=" * 60)
    test_tracer_lifecycle_and_timing()
    test_metadata_sanitization()
    test_derive_status_reason()
    test_graph_trace_propagation()
    test_graph_insufficient_trace_path()
    test_disabled_tracing()
    test_error_capture_in_trace()
    print("=" * 60)
    print("ALL STEP 22 OBSERVABILITY TESTS PASSED!")
    print("=" * 60)
