"""
FastAPI routing endpoints for PolicyPilot.
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import HealthResponse, QuestionRequest, QuestionResponse, CitationItem
from app.api.pipeline import pipeline_manager

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Confirms that the FastAPI server is running without loading heavy ML models.",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="PolicyPilot",
    )


@router.post(
    "/ask",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Query PolicyPilot",
    description="Executes the PolicyPilot workflow on an enterprise policy query.",
)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    cleaned_query = request.question.strip() if request.question else ""
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    try:
        graph = pipeline_manager.get_graph()
        workflow_input = {
            "original_query": cleaned_query,
            "evidence": [],
            "trace_id": request.trace_id,
            "enable_tracing": request.enable_tracing,
        }
        prediction = graph.invoke(workflow_input)

        final_status = prediction.get("status", "error")
        final_answer = prediction.get("final_answer", "")
        error_msg = prediction.get("error")

        if final_status == "error" and error_msg:
            # Handle workflow error safely without exposing raw trace/secrets
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Workflow execution failed: {error_msg}",
            )

        resp_result = prediction.get("response_result") or {}
        raw_citations = resp_result.get("citations", [])
        citations: list[CitationItem] = []
        for c in raw_citations:
            if isinstance(c, dict) and "citation" in c:
                citations.append(
                    CitationItem(
                        citation=c["citation"],
                        chunk_id=c.get("chunk_id"),
                        document_id=c.get("document_id"),
                        document_name=c.get("document_name"),
                        page_number=c.get("page_number"),
                    )
                )

        trace_data = prediction.get("trace")
        # Strip internal Python objects if present
        if isinstance(trace_data, dict):
            trace_data = {k: v for k, v in trace_data.items() if not k.startswith("_")}

        return QuestionResponse(
            question=cleaned_query,
            answer=final_answer or "No answer could be determined.",
            status=final_status,
            citations=citations,
            trace_id=prediction.get("trace_id"),
            trace=trace_data if request.enable_tracing else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Prevent leaking stack trace / internal details
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error processing request: {str(e)}",
        )
