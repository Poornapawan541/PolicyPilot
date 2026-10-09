import json
import re
import urllib.request
import urllib.error
from typing import Any, Optional


class ReasoningAgent:
    """
    Reasoning Agent (Step 14).

    Derives grounded reasoning conclusions using ONLY verified evidence that
    has passed the Evidence Verification Agent gate.

    Does not retrieve documents, verify evidence independently, invent facts,
    use outside knowledge, or generate citations.
    """

    ALLOWED_STATUSES = {"grounded", "insufficient", "conflicting"}

    def __init__(
        self,
        model_name: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def build_prompt(self, query: str, verified_evidence: list[dict]) -> str:
        """
        Builds a strict grounded reasoning prompt instructing the model to derive
        a conclusion using ONLY the supplied verified evidence.
        """
        snippets = []
        for item in verified_evidence:
            cid = item.get("chunk_id", "unknown")
            page = item.get("page_number", "unknown")
            doc = item.get("document_name") or item.get("document_id") or ""
            doc_str = f" | Document: {doc}" if doc else ""
            text = item.get("text", "").strip()
            snippets.append(f"[Chunk ID: {cid} | Page: {page}{doc_str}]\n{text}")

        formatted_evidence = "\n\n".join(snippets)

        prompt = f"""You are a grounded reasoning component for an enterprise compliance system.
Your ONLY role is to derive a conclusion that directly answers the user's question using ONLY the supplied verified evidence.

CRITICAL RULES:
1. Use ONLY the supplied verified evidence.
2. Do NOT use outside knowledge, general world knowledge, or assumptions.
3. Do NOT invent missing information, unstated policies, or extra conditions.
4. Do NOT contradict the evidence.
5. In "evidence_used", include ONLY the chunk_id and page_number of items from the verified evidence that directly support your conclusion. Do NOT fabricate chunk IDs.
6. Output strictly valid JSON matching the exact schema below, with no explanation, markdown formatting, or text outside JSON.

Exact JSON schema:
{{
  "original_query": "{query}",
  "status": "grounded",
  "conclusion": "<concise, direct answer based strictly on the verified evidence>",
  "reason": "<short explanation of how the evidence supports the conclusion>",
  "evidence_used": [
    {{
      "chunk_id": "<exact chunk_id from verified evidence>",
      "page_number": <exact page_number integer>
    }}
  ]
}}

User Question:
{query}

Verified Evidence:
{formatted_evidence}

JSON:"""
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP status {response.status}")
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to communicate with Ollama at {self.ollama_url}: {e}")

    def _validate_inputs(
        self,
        original_query: str,
        verification_result: dict,
        evidence: list[dict],
    ) -> None:
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must be a non-empty string.")

        if not isinstance(verification_result, dict):
            raise ValueError("verification_result must be a dictionary.")

        if "is_sufficient" not in verification_result:
            raise ValueError("verification_result must contain 'is_sufficient'.")

        if not isinstance(verification_result["is_sufficient"], bool):
            raise ValueError("'is_sufficient' in verification_result must be a boolean.")

        if "status" not in verification_result:
            raise ValueError("verification_result must contain 'status'.")

        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list.")

    def _parse_and_validate(
        self,
        raw_response: str,
        original_query: str,
        verified_evidence: list[dict],
    ) -> dict:
        """
        Parses JSON from Ollama response and validates schema, required fields,
        and evidence references.
        """
        if not raw_response or not raw_response.strip():
            raise ValueError("Model returned an empty response.")

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON from response: {cleaned}") from e
            else:
                raise ValueError(f"Failed to parse JSON from response: {cleaned}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        # Check required fields
        required_fields = ["status", "conclusion", "reason", "evidence_used"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in reasoning output: '{field}'")

        if data["status"] not in self.ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid 'status': {data['status']}. Allowed values: {self.ALLOWED_STATUSES}"
            )

        if not isinstance(data["conclusion"], str) or not data["conclusion"].strip():
            raise ValueError("'conclusion' must be a non-empty string for grounded result.")

        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise ValueError("'reason' must be a non-empty string.")

        if not isinstance(data["evidence_used"], list):
            raise ValueError("'evidence_used' must be a list.")

        if not data["evidence_used"]:
            raise ValueError("'evidence_used' must not be empty for grounded status.")

        valid_chunks_map = {item["chunk_id"]: item.get("page_number") for item in verified_evidence}

        validated_evidence_used = []
        for idx, item in enumerate(data["evidence_used"]):
            if not isinstance(item, dict):
                raise ValueError(f"evidence_used item at index {idx} must be a dictionary.")
            chunk_id = item.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise ValueError(f"evidence_used item at index {idx} missing valid 'chunk_id'.")

            if chunk_id not in valid_chunks_map:
                raise ValueError(
                    f"Hallucinated chunk ID '{chunk_id}' in evidence_used not in verified evidence."
                )

            page_number = item.get("page_number", valid_chunks_map[chunk_id])
            if not isinstance(page_number, int):
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    page_number = valid_chunks_map[chunk_id]

            entry = {
                "chunk_id": chunk_id,
                "page_number": page_number,
            }
            source_chunk = next((e for e in verified_evidence if e.get("chunk_id") == chunk_id), None)
            if source_chunk:
                if source_chunk.get("document_id"):
                    entry["document_id"] = source_chunk["document_id"]
                if source_chunk.get("document_name"):
                    entry["document_name"] = source_chunk["document_name"]

            validated_evidence_used.append(entry)

        return {
            "original_query": original_query,
            "status": "grounded",
            "conclusion": data["conclusion"].strip(),
            "reason": data["reason"].strip(),
            "evidence_used": validated_evidence_used,
        }

    def reason(
        self,
        original_query: str,
        verification_result: dict,
        evidence: list[dict],
    ) -> dict:
        """
        Derives a grounded reasoning conclusion given:
        1. original_query
        2. verification_result (from EvidenceVerificationAgent)
        3. evidence (from RetrievalAgent)

        Verification gate ensures unverified or conflicting evidence does not produce
        a grounded conclusion.
        """
        self._validate_inputs(original_query, verification_result, evidence)
        cleaned_query = original_query.strip()

        # Deterministic check for empty evidence list
        if not evidence:
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "conclusion": None,
                "reason": "No evidence was provided.",
                "evidence_used": [],
            }

        # Verification Gate 1: Check conflicting status
        if verification_result.get("status") == "conflicting":
            return {
                "original_query": cleaned_query,
                "status": "conflicting",
                "conclusion": None,
                "reason": verification_result.get(
                    "reason", "The supplied evidence contains conflicting information."
                ),
                "evidence_used": [],
                "conflicts": verification_result.get("conflicts", []),
            }

        # Verification Gate 2: Check is_sufficient
        if not verification_result.get("is_sufficient", False):
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "conclusion": None,
                "reason": verification_result.get(
                    "reason", "The available evidence is insufficient to answer the question."
                ),
                "evidence_used": [],
            }

        # Extract verified supporting evidence
        supporting_refs = verification_result.get("supporting_evidence", [])
        evidence_by_id = {item.get("chunk_id"): item for item in evidence if isinstance(item, dict)}

        if supporting_refs:
            # Check for verified evidence consistency: references must exist in supplied evidence
            verified_evidence = []
            for ref in supporting_refs:
                cid = ref.get("chunk_id") if isinstance(ref, dict) else None
                if not cid or cid not in evidence_by_id:
                    # Verified evidence mismatch: verification references chunks not in evidence
                    return {
                        "original_query": cleaned_query,
                        "status": "insufficient",
                        "conclusion": None,
                        "reason": f"Verification referenced chunk '{cid}' which was not found in supplied evidence.",
                        "evidence_used": [],
                    }
                verified_evidence.append(evidence_by_id[cid])
        else:
            # If is_sufficient was True but supporting_evidence was empty, fallback safely to evidence
            verified_evidence = evidence

        # Call LLM for grounded conclusion
        prompt = self.build_prompt(cleaned_query, verified_evidence)
        try:
            raw_response = self._call_ollama(prompt)
            return self._parse_and_validate(raw_response, cleaned_query, verified_evidence)
        except Exception as e:
            # Safe failure: never fail open
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "conclusion": None,
                "reason": f"Reasoning failed due to parsing/validation error: {str(e)}",
                "evidence_used": [],
            }
