"""
Pydantic request and response schemas for PolicyPilot API.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status")
    version: str = Field(default="1.0.0", description="Application version")
    service: str = Field(default="PolicyPilot", description="Service name")


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The natural language question to ask PolicyPilot",
        examples=["How many days of annual leave do employees receive per year?"],
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Optional custom client trace identifier",
    )
    enable_tracing: Optional[bool] = Field(
        default=True,
        description="Whether to capture workflow execution stages and timing",
    )


class CitationItem(BaseModel):
    citation: str = Field(..., description="Formatted bracket citation label, e.g. [Page 2]")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk identifier")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    document_name: Optional[str] = Field(default=None, description="Source document name")
    page_number: Optional[int] = Field(default=None, description="Page number within source document")


class QuestionResponse(BaseModel):
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Synthesized answer or explanation")
    status: str = Field(..., description="Decision status: grounded, insufficient, or conflicting")
    citations: list[CitationItem] = Field(default_factory=list, description="Citations supporting answer")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for this execution")
    trace: Optional[dict[str, Any]] = Field(default=None, description="Stage timing and execution metadata")
