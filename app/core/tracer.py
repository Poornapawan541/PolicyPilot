"""
Local-first, lightweight observability and tracing for PolicyPilot.

Provides structured trace IDs, workflow stage timing, safe metadata capture,
status rationale explanations, and zero-leak payload sanitization.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


class WorkflowTracer:
    """
    In-memory workflow tracer for individual PolicyPilot query executions.
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        service_name: str = "policypilot",
    ):
        if enabled is None:
            # Check environment variable, default to True if unset
            env_val = os.getenv("POLICYPILOT_TRACING_ENABLED", "true").strip().lower()
            self.enabled = env_val not in ("false", "0", "no", "off")
        else:
            self.enabled = bool(enabled)

        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        self.service_name = service_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.stages: list[dict[str, Any]] = []
        self._active_spans: dict[str, float] = {}
        self.error: Optional[str] = None
        self.final_status: Optional[str] = None
        self.status_reason: Optional[str] = None

        if self.enabled:
            self.start_time = time.time()

    def start_stage(self, stage_name: str) -> None:
        """Records start of a workflow stage."""
        if not self.enabled:
            return
        self._active_spans[stage_name] = time.time()

    def end_stage(
        self,
        stage_name: str,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Records end of a workflow stage with sanitized metadata and duration."""
        if not self.enabled:
            return

        start = self._active_spans.pop(stage_name, None)
        end = time.time()
        duration_ms = round((end - start) * 1000, 2) if start else 0.0

        sanitized_meta = self._sanitize_metadata(metadata or {})

        stage_record = {
            "stage": stage_name,
            "start_time": datetime.fromtimestamp(start or end, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "metadata": sanitized_meta,
            "error": error,
        }
        self.stages.append(stage_record)

    def record_decision(
        self,
        final_status: str,
        reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Records the final status classification and explanatory reason."""
        if not self.enabled:
            return
        self.final_status = final_status
        self.status_reason = reason
        if error:
            self.error = error

    def finish(self) -> dict[str, Any]:
        """Finalizes and returns the complete trace summary."""
        if not self.enabled:
            return {
                "trace_id": self.trace_id,
                "enabled": False,
            }

        self.end_time = time.time()
        total_duration_ms = (
            round((self.end_time - (self.start_time or self.end_time)) * 1000, 2)
        )

        return {
            "trace_id": self.trace_id,
            "service": self.service_name,
            "enabled": True,
            "start_time": datetime.fromtimestamp(self.start_time or self.end_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat(),
            "total_duration_ms": total_duration_ms,
            "final_status": self.final_status,
            "status_reason": self.status_reason,
            "error": self.error,
            "stages": self.stages,
        }

    @staticmethod
    def _sanitize_metadata(data: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitizes stage metadata to avoid logging sensitive user credentials,
        API keys, or massive text chunks.
        """
        sanitized = {}
        sensitive_keys = {"password", "secret", "token", "api_key", "authorization", "auth"}

        for k, v in data.items():
            k_lower = k.lower()
            if any(s in k_lower for s in sensitive_keys):
                sanitized[k] = "[REDACTED]"
            elif k == "evidence" and isinstance(v, list):
                # Preserve chunk identifiers, document ID, page number, and score without dumping full text
                sanitized_chunks = []
                for chunk in v:
                    if isinstance(chunk, dict):
                        chunk_summary = {
                            "chunk_id": chunk.get("chunk_id"),
                            "document_id": chunk.get("document_id"),
                            "document_name": chunk.get("document_name"),
                            "page_number": chunk.get("page_number"),
                        }
                        if "score" in chunk:
                            chunk_summary["score"] = round(chunk["score"], 4) if isinstance(chunk["score"], (int, float)) else chunk["score"]
                        sanitized_chunks.append(chunk_summary)
                sanitized["evidence_chunks"] = sanitized_chunks
                sanitized["evidence_count"] = len(sanitized_chunks)
            elif k == "text" and isinstance(v, str) and len(v) > 200:
                sanitized[k] = f"{v[:100]}... [truncated {len(v)} chars]"
            else:
                sanitized[k] = v

        return sanitized


def derive_status_reason(
    status: Optional[str],
    verification_result: Optional[dict[str, Any]],
    reasoning_result: Optional[dict[str, Any]],
    error: Optional[str] = None,
) -> str:
    """
    Produces a human-readable explanation of why a query resolved
    to grounded, insufficient, conflicting, or error.
    """
    if error:
        return f"Workflow halted due to error: {error}"

    ver_status = (verification_result or {}).get("status") if isinstance(verification_result, dict) else None
    conflicts = (verification_result or {}).get("conflicts", []) if isinstance(verification_result, dict) else []

    if status == "conflicting" or ver_status == "conflicting":
        num_conflicts = len(conflicts)
        return (
            f"Cross-document contradictions detected ({num_conflicts} conflicting claims). "
            "Evidence could not be reconciled via explicit supersession rules."
        )

    if status == "insufficient" or ver_status == "insufficient":
        missing_aspects = (verification_result or {}).get("missing_aspects", [])
        if missing_aspects:
            aspect_str = ", ".join(missing_aspects[:3])
            return f"Retrieved documents do not provide adequate evidence. Missing aspects: {aspect_str}."
        return "Retrieved documents do not contain sufficient verified evidence to support the claim."

    if status == "grounded" or ver_status == "sufficient":
        evidence_used = (reasoning_result or {}).get("evidence_used", []) if isinstance(reasoning_result, dict) else []
        count = len(evidence_used)
        return f"Query answered successfully with {count} verified supporting evidence chunk(s)."

    return f"Query completed with status '{status}'."
