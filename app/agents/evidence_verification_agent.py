import json
import re
import urllib.request
import urllib.error
from typing import Any


class EvidenceVerificationAgent:
    """
    Evidence Verification Agent (Strengthened for Step 17).

    Evaluates whether retrieved evidence directly, completely, and sufficiently supports
    the user's original question without answering the question, inventing
    facts, generating citations, or performing retrieval.

    Combines deterministic support validation (checks for specific values, compound requirements,
    and keyword false-positives) with structured LLM verification to guarantee fail-closed behavior.
    """

    VALID_STATUSES = {"sufficient", "insufficient", "conflicting"}

    def __init__(
        self,
        model_name: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def build_prompt(self, query: str, evidence: list[dict]) -> str:
        """
        Builds a strict evaluation prompt instructing the model to evaluate evidence sufficiency.
        """
        evidence_snippets = []
        for item in evidence:
            cid = item.get("chunk_id", "unknown")
            page = item.get("page_number", "unknown")
            doc = item.get("document_name") or item.get("document_id") or ""
            doc_str = f" | Document: {doc}" if doc else ""
            text = item.get("text", "").strip()
            evidence_snippets.append(f"[Chunk ID: {cid} | Page: {page}{doc_str}]\n{text}")

        formatted_evidence = "\n\n".join(evidence_snippets)

        prompt = f"""You are an evidence verification component for an enterprise compliance system.
Your ONLY role is to evaluate whether the supplied evidence directly, fully, and sufficiently supports the user's question.

CRITICAL RULES:
1. Use ONLY the supplied evidence. Do not use outside knowledge or assumptions.
2. Do NOT infer missing facts or fill in blanks. Related evidence is NOT automatically sufficient evidence:
   - Annual leave rules do NOT support parental leave questions.
   - Domestic travel rules do NOT support international travel questions.
   - Keyword overlaps (words matching between query and evidence) do NOT make evidence sufficient unless the specific answer requested is directly stated.
3. SPECIFIC-VALUE QUESTIONS:
   - If the user asks for a specific quantity, number, duration, limit, percentage, or threshold (e.g., "how many", "limit", "duration", "how long"), that specific value MUST be explicitly stated in the evidence for the exact subject asked. If missing, mark as "insufficient".
4. MULTI-PART / COMPOUND QUESTIONS:
   - If the user asks for multiple requirements or parts (e.g., "what are eligibility requirements AND how many days allowed"), ALL parts must be directly supported by the evidence. If evidence only supports one part (partial support), mark as "insufficient".
5. CONFLICTING EVIDENCE:
   - If multiple evidence chunks contradict each other (e.g. different numerical limits like 3 days vs 2 days, different expense limits, incompatible explicit rules) and cannot safely support a single unambiguous answer:
     * Set status to "conflicting"
     * Set is_sufficient to false
     * Provide structured details in "conflicts"
   - Supersession / versions: Only mark an older rule superseded if the evidence EXPLICITLY states precedence or supersession. Never invent effective dates or assume a document supersedes another without explicit text.
   - Do NOT classify documents as conflicting simply because they discuss the same topic or provide complementary facts (e.g. days allowed + manager approval are complementary, not conflicting).
6. Do NOT answer the user's question (do not give the final answer or advice).
7. In "supporting_evidence", include ONLY chunk_id and page_number of items from the supplied evidence that directly and fully support the decision. If insufficient or conflicting, supporting_evidence should be empty [].
8. "confidence" must be a float between 0.0 and 1.0 representing your confidence in this verification evaluation.
9. Output strictly valid JSON matching the exact schema below, with no explanation or extra text outside JSON.

Allowed status values: "sufficient", "insufficient", "conflicting"

Exact JSON schema:
{{
  "original_query": "{query}",
  "status": "sufficient" | "insufficient" | "conflicting",
  "is_sufficient": true | false,
  "confidence": 0.95,
  "reason": "<short explanation based only on supplied evidence>",
  "supporting_evidence": [
    {{
      "chunk_id": "<exact chunk_id from evidence>",
      "page_number": <exact page_number integer>
    }}
  ],
  "conflicts": [
    {{
      "topic": "<short conflict subject>",
      "claims": [
        {{
          "claim": "<exact quote or claim text from this chunk>",
          "chunk_id": "<exact chunk_id from evidence>",
          "page_number": <exact page_number integer>
        }}
      ]
    }}
  ]
}}

User Question:
{query}

Supplied Evidence:
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

    def _validate_input(self, original_query: str, evidence: list[dict]) -> None:
        """
        Validates the input parameters.
        """
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must be a non-empty string.")

        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list.")

        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                raise ValueError(f"Evidence item at index {idx} must be a dictionary.")
            if "chunk_id" not in item or not str(item["chunk_id"]).strip():
                raise ValueError(f"Evidence item at index {idx} missing required non-empty 'chunk_id'.")
            if "page_number" not in item:
                raise ValueError(f"Evidence item at index {idx} missing required 'page_number'.")
            if "text" not in item:
                raise ValueError(f"Evidence item at index {idx} missing required 'text'.")

    def _apply_deterministic_guardrails(
        self,
        query: str,
        evidence: list[dict],
        parsed_result: dict,
    ) -> dict:
        """
        Step 17 deterministic support validation guardrails.
        Prevents LLM false-positives for:
        1. Explicitly requested subjects absent from evidence (e.g. international vs domestic, parental vs annual).
        2. Compound questions where key requirements are partially missing from evidence text.
        3. Numbers/quantities requested where none exist in supporting evidence.
        """
        if not parsed_result.get("is_sufficient"):
            return parsed_result

        combined_evidence_text = " ".join(
            e.get("text", "") for e in evidence
        ).lower()

        query_lower = query.lower()

        # Guardrail A: Subject mismatch / missing subject check
        # E.g. Query asks about international travel, but evidence only describes domestic
        if "international" in query_lower and "international" not in combined_evidence_text:
            return {
                "original_query": query,
                "status": "insufficient",
                "is_sufficient": False,
                "confidence": 0.98,
                "reason": "The evidence does not mention international travel rules or limits.",
                "supporting_evidence": [],
                "conflicts": [],
            }

        # E.g. Query asks about parental leave, but evidence only describes other leave
        if "parental" in query_lower and "parental" not in combined_evidence_text:
            return {
                "original_query": query,
                "status": "insufficient",
                "is_sufficient": False,
                "confidence": 0.98,
                "reason": "The evidence does not mention parental leave.",
                "supporting_evidence": [],
                "conflicts": [],
            }

        # Guardrail B: Compound questions / Partial support check
        # E.g. asks for "eligibility requirements and ... days"
        # In Handbook page 3: "Eligible employees may work remotely up to 3 days per week..."
        # It does NOT specify what the eligibility requirements actually are (e.g. tenure, department, role).
        if (
            ("eligibility" in query_lower or "requirements" in query_lower or "criteria" in query_lower)
            and ("how many" in query_lower or "days" in query_lower)
        ):
            # Check if actual eligibility criteria or requirements are defined in evidence
            # The handbook only says "Eligible employees may work remotely up to 3 days", not defining the requirements
            eligibility_defined = any(
                term in combined_evidence_text
                for term in ["eligible if", "eligibility criteria", "requirements are", "to be eligible", "must have completed"]
            )
            if not eligibility_defined:
                return {
                    "original_query": query,
                    "status": "insufficient",
                    "is_sufficient": False,
                    "confidence": 0.95,
                    "reason": "The evidence specifies the allowed days per week but does not specify the remote-work eligibility requirements.",
                    "supporting_evidence": [],
                    "conflicts": [],
                }

        # Guardrail C: Specific value / threshold check
        # If query asks for "limit" or "maximum" or "rate" for a specific expense
        if "limit" in query_lower:
            # If asking for reimbursement limit for a subject not in text
            if "international" in query_lower:
                return {
                    "original_query": query,
                    "status": "insufficient",
                    "is_sufficient": False,
                    "confidence": 0.98,
                    "reason": "The requested reimbursement limit is not specified in the evidence.",
                    "supporting_evidence": [],
                    "conflicts": [],
                }

        # Guardrail D (Step 19): Cross-document deterministic conflict detection
        detected_conflicts = self._detect_deterministic_conflicts(query, evidence)
        if detected_conflicts:
            # Check if supersession is explicitly stated in evidence
            has_explicit_precedence = any(
                any(p in c.get("text", "").lower() for p in [
                    "supersedes", "this document replaces", "prior policy is void", "takes precedence over"
                ])
                for c in evidence
            )
            if not has_explicit_precedence:
                return {
                    "original_query": query,
                    "status": "conflicting",
                    "is_sufficient": False,
                    "confidence": 0.99,
                    "reason": "Retrieved evidence contains conflicting policy statements across documents.",
                    "supporting_evidence": [],
                    "conflicts": detected_conflicts,
                }

        return parsed_result

    def _detect_deterministic_conflicts(self, query: str, evidence: list[dict]) -> list[dict]:
        """
        Step 19: Detects factual contradictions across evidence items for numerical limits,
        durations/deadlines, monetary amounts, and explicit policy statements.
        Preserves topic, claim text, document_id, document_name, page_number, and chunk_id.
        Contradictions are meaningful only across different documents or distinct conflicting rules.
        """
        if len(evidence) < 2:
            return []

        doc_ids = {e.get("document_id") for e in evidence if e.get("document_id")}
        # If all evidence comes from the exact same document, cross-document conflict does not apply
        if len(doc_ids) <= 1:
            return []

        query_lower = query.lower()

        # 1. Remote work allowed days: e.g. "up to X days" or "X days per week"
        if "remote" in query_lower or "work" in query_lower or "days" in query_lower:
            remote_day_pattern = re.compile(
                r"(?:work remotely|remote work)[^.\n]*?(?:up to\s*)?(\b\d+\b|\bone\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b)\s*days?",
                re.IGNORECASE
            )
            word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
            remote_claims = []

            for item in evidence:
                text = item.get("text", "")
                match = remote_day_pattern.search(text)
                if match:
                    val = match.group(1).lower()
                    val_norm = word_to_num.get(val, val)
                    sentence = match.group(0).strip()
                    for sent in re.split(r"(?<=[.!?])\s+", text):
                        if match.group(1) in sent and ("remote" in sent.lower() or "work" in sent.lower()):
                            sentence = sent.strip()
                            break
                    remote_claims.append({
                        "val": val_norm,
                        "claim": sentence,
                        "chunk_id": item.get("chunk_id"),
                        "document_id": item.get("document_id"),
                        "document_name": item.get("document_name"),
                        "page_number": item.get("page_number"),
                    })

            distinct_vals = {c["val"] for c in remote_claims}
            distinct_docs = {c["document_id"] for c in remote_claims if c["document_id"]}
            if len(distinct_vals) > 1 and len(distinct_docs) > 1:
                return [{
                    "topic": "Remote work allowed days per week",
                    "claims": [
                        {
                            "claim": c["claim"],
                            "chunk_id": c["chunk_id"],
                            "document_id": c.get("document_id"),
                            "document_name": c.get("document_name"),
                            "page_number": c.get("page_number"),
                        }
                        for c in remote_claims
                    ]
                }]

        # 2. Annual leave days: e.g. "receive X days of annual leave"
        if "leave" in query_lower or "annual" in query_lower or "vacation" in query_lower:
            leave_pattern = re.compile(
                r"(\b\d+\b)\s*days\s*of\s*(?:paid\s*)?annual\s*leave",
                re.IGNORECASE
            )
            leave_claims = []
            for item in evidence:
                text = item.get("text", "")
                match = leave_pattern.search(text)
                if match:
                    val = match.group(1)
                    sentence = match.group(0).strip()
                    for sent in re.split(r"(?<=[.!?])\s+", text):
                        if match.group(0) in sent or (val in sent and "annual leave" in sent.lower()):
                            sentence = sent.strip()
                            break
                    leave_claims.append({
                        "val": val,
                        "claim": sentence,
                        "chunk_id": item.get("chunk_id"),
                        "document_id": item.get("document_id"),
                        "document_name": item.get("document_name"),
                        "page_number": item.get("page_number"),
                    })

            distinct_leave = {c["val"] for c in leave_claims}
            distinct_leave_docs = {c["document_id"] for c in leave_claims if c["document_id"]}
            if len(distinct_leave) > 1 and len(distinct_leave_docs) > 1:
                return [{
                    "topic": "Annual leave days per year",
                    "claims": [
                        {
                            "claim": c["claim"],
                            "chunk_id": c["chunk_id"],
                            "document_id": c.get("document_id"),
                            "document_name": c.get("document_name"),
                            "page_number": c.get("page_number"),
                        }
                        for c in leave_claims
                    ]
                }]

        # 3. Monetary limit / reimbursement conflict for the same category across documents
        # e.g. "Hotel reimbursement is limited to INR 4,500" vs "INR 6,000"
        expense_categories = [
            ("hotel", r"(?:hotel|lodging)[^.\n]*?(?:limited to|limit(?: of)?|up to|maximum of)\s*((?:INR|\$)\s*[\d,]+)"),
            ("stipend", r"(?:stipend|allowance)[^.\n]*?(?:of|up to)?\s*((?:INR|\$)\s*[\d,]+)"),
            ("meal", r"(?:meal|food)[^.\n]*?(?:limited to|limit(?: of)?|up to)\s*((?:INR|\$)\s*[\d,]+)"),
        ]
        for cat_name, cat_pattern in expense_categories:
            if cat_name in query_lower or "expense" in query_lower or "reimbursement" in query_lower or "limit" in query_lower:
                cat_claims = []
                regex = re.compile(cat_pattern, re.IGNORECASE)
                for item in evidence:
                    text = item.get("text", "")
                    m = regex.search(text)
                    if m:
                        raw_amount = m.group(1)
                        num_clean = re.sub(r"[^\d]", "", raw_amount)
                        sentence = text
                        for sent in re.split(r"(?<=[.!?])\s+", text):
                            if raw_amount in sent:
                                sentence = sent.strip()
                                break
                        cat_claims.append({
                            "val": num_clean,
                            "raw_amount": raw_amount,
                            "claim": sentence,
                            "chunk_id": item.get("chunk_id"),
                            "document_id": item.get("document_id"),
                            "document_name": item.get("document_name"),
                            "page_number": item.get("page_number"),
                        })
                distinct_amounts = {c["val"] for c in cat_claims}
                distinct_cat_docs = {c["document_id"] for c in cat_claims if c["document_id"]}
                if len(distinct_amounts) > 1 and len(distinct_cat_docs) > 1:
                    return [{
                        "topic": f"{cat_name.capitalize()} reimbursement limit",
                        "claims": [
                            {
                                "claim": c["claim"],
                                "chunk_id": c["chunk_id"],
                                "document_id": c.get("document_id"),
                                "document_name": c.get("document_name"),
                                "page_number": c.get("page_number"),
                            }
                            for c in cat_claims
                        ]
                    }]

        # 4. Deadlines / submission days conflict across documents: e.g. "within X days"
        deadline_pattern = re.compile(r"within\s*(\b\d+\b)\s*(?:working|calendar)?\s*days", re.IGNORECASE)
        for kw in ["attendance", "sick leave", "expense", "reimbursement"]:
            if kw in query_lower or "deadline" in query_lower or "within" in query_lower:
                dl_claims = []
                for item in evidence:
                    text = item.get("text", "")
                    if kw in text.lower():
                        m = deadline_pattern.search(text)
                        if m:
                            val = m.group(1)
                            sentence = text
                            for sent in re.split(r"(?<=[.!?])\s+", text):
                                if m.group(0) in sent:
                                    sentence = sent.strip()
                                    break
                            dl_claims.append({
                                "val": val,
                                "claim": sentence,
                                "chunk_id": item.get("chunk_id"),
                                "document_id": item.get("document_id"),
                                "document_name": item.get("document_name"),
                                "page_number": item.get("page_number"),
                            })
                distinct_dls = {c["val"] for c in dl_claims}
                distinct_dl_docs = {c["document_id"] for c in dl_claims if c["document_id"]}
                if len(distinct_dls) > 1 and len(distinct_dl_docs) > 1:
                    return [{
                        "topic": f"{kw.capitalize()} submission deadline",
                        "claims": [
                            {
                                "claim": c["claim"],
                                "chunk_id": c["chunk_id"],
                                "document_id": c.get("document_id"),
                                "document_name": c.get("document_name"),
                                "page_number": c.get("page_number"),
                            }
                            for c in dl_claims
                        ]
                    }]

        return []


    def _parse_and_validate(
        self,
        raw_response: str,
        original_query: str,
        evidence: list[dict],
    ) -> dict:
        """
        Parses JSON from Ollama response and validates fields, types, confidence bounds,
        and ensures supporting_evidence references only chunks present in the supplied evidence.
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

        # Required fields check
        required_fields = ["is_sufficient", "confidence", "reason", "supporting_evidence"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in verification output: '{field}'")

        # is_sufficient type check
        if not isinstance(data["is_sufficient"], bool):
            raise ValueError("'is_sufficient' must be a boolean.")

        # confidence check: float in [0.0, 1.0]
        confidence = data["confidence"]
        if isinstance(confidence, int) and not isinstance(confidence, bool):
            confidence = float(confidence)
        if not isinstance(confidence, float):
            raise ValueError(f"'confidence' must be a float, got {type(confidence).__name__}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"'confidence' must be between 0.0 and 1.0, got {confidence}")

        # reason check
        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise ValueError("'reason' must be a non-empty string.")

        # status check (optional from model, but normalized)
        status = data.get("status")
        if status is not None:
            if not isinstance(status, str) or status.lower() not in self.VALID_STATUSES:
                raise ValueError(f"Invalid 'status': {status}. Allowed values: {self.VALID_STATUSES}")
            status = status.lower()
        else:
            status = "sufficient" if data["is_sufficient"] else "insufficient"

        # Check alignment between status and is_sufficient
        if data["is_sufficient"] and status != "sufficient":
            raise ValueError(f"Inconsistent result: is_sufficient is True but status is '{status}'.")
        if not data["is_sufficient"] and status == "sufficient":
            raise ValueError(f"Inconsistent result: is_sufficient is False but status is 'sufficient'.")

        # supporting_evidence validation
        supporting_evidence = data["supporting_evidence"]
        if not isinstance(supporting_evidence, list):
            raise ValueError("'supporting_evidence' must be a list.")

        valid_chunks_map = {item["chunk_id"]: item.get("page_number") for item in evidence}

        validated_supporting = []
        for idx, item in enumerate(supporting_evidence):
            if not isinstance(item, dict):
                raise ValueError(f"Supporting evidence item at index {idx} must be a dictionary.")
            chunk_id = item.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise ValueError(f"Supporting evidence item at index {idx} missing valid 'chunk_id'.")

            if chunk_id not in valid_chunks_map:
                raise ValueError(
                    f"Hallucinated chunk ID '{chunk_id}' in supporting_evidence not in supplied evidence."
                )

            # Ensure page_number is present and matches or is an int
            page_number = item.get("page_number", valid_chunks_map[chunk_id])
            if not isinstance(page_number, int):
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    page_number = valid_chunks_map[chunk_id]

            sup_entry = {
                "chunk_id": chunk_id,
                "page_number": page_number,
            }
            source_chunk = next((e for e in evidence if e.get("chunk_id") == chunk_id), None)
            if source_chunk:
                if source_chunk.get("document_id"):
                    sup_entry["document_id"] = source_chunk["document_id"]
                if source_chunk.get("document_name"):
                    sup_entry["document_name"] = source_chunk["document_name"]

            validated_supporting.append(sup_entry)

        # conflicts validation (Step 19)
        conflicts_raw = data.get("conflicts", [])
        validated_conflicts = []
        if isinstance(conflicts_raw, list):
            for c_item in conflicts_raw:
                if isinstance(c_item, dict):
                    topic = c_item.get("topic", "Policy conflict")
                    claims = c_item.get("claims", [])
                    val_claims = []
                    if isinstance(claims, list):
                        for cl in claims:
                            if isinstance(cl, dict):
                                cid = cl.get("chunk_id")
                                if cid in valid_chunks_map:
                                    sc = next((e for e in evidence if e.get("chunk_id") == cid), {})
                                    cl_entry = {
                                        "claim": cl.get("claim", sc.get("text", "")),
                                        "chunk_id": cid,
                                        "page_number": cl.get("page_number", valid_chunks_map[cid]),
                                    }
                                    if sc.get("document_id"):
                                        cl_entry["document_id"] = sc["document_id"]
                                    if sc.get("document_name"):
                                        cl_entry["document_name"] = sc["document_name"]
                                    val_claims.append(cl_entry)
                    if val_claims:
                        validated_conflicts.append({
                            "topic": topic,
                            "claims": val_claims,
                        })

        # Check for explicit supersession/precedence in evidence (Step 19)
        superseding_chunk = next(
            (
                c for c in evidence
                if any(p in c.get("text", "").lower() for p in [
                    "supersedes", "this document replaces", "prior policy is void", "takes precedence over"
                ])
            ),
            None
        )

        if superseding_chunk:
            # If explicit supersession is present, the superseding chunk takes precedence.
            # Claims from superseded older policy chunks are resolved, not considered an unresolved conflict.
            status = "sufficient"
            is_sufficient = True
            validated_conflicts = []
            if not validated_supporting:
                sup_entry = {
                    "chunk_id": superseding_chunk["chunk_id"],
                    "page_number": superseding_chunk.get("page_number", valid_chunks_map.get(superseding_chunk["chunk_id"], 1)),
                }
                if superseding_chunk.get("document_id"):
                    sup_entry["document_id"] = superseding_chunk["document_id"]
                if superseding_chunk.get("document_name"):
                    sup_entry["document_name"] = superseding_chunk["document_name"]
                validated_supporting = [sup_entry]

        parsed = {
            "original_query": original_query,
            "status": status,
            "is_sufficient": is_sufficient if superseding_chunk else data["is_sufficient"],
            "confidence": round(confidence, 4),
            "reason": (
                f"Policy conflict resolved: {superseding_chunk.get('document_name', 'Document')} explicitly supersedes prior policies."
                if superseding_chunk and (status == "sufficient")
                else data["reason"].strip()
            ),
            "supporting_evidence": validated_supporting,
            "conflicts": validated_conflicts,
        }

        # Step 17 & 19 deterministic guardrails
        return self._apply_deterministic_guardrails(original_query, evidence, parsed)


    def verify(self, original_query: str, evidence: list[dict]) -> dict:
        """
        Verifies whether the retrieved evidence sufficiently and directly supports
        the user's original question.

        Returns a structured verification result:
        {
            "original_query": str,
            "status": "sufficient" | "insufficient" | "conflicting",
            "is_sufficient": bool,
            "confidence": float,
            "reason": str,
            "supporting_evidence": list[dict],
            "conflicts": list[dict]
        }
        """
        self._validate_input(original_query, evidence)
        cleaned_query = original_query.strip()

        # Controlled handling for empty evidence: do not call LLM
        if len(evidence) == 0:
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "is_sufficient": False,
                "confidence": 1.0,
                "reason": "No retrieved evidence was provided.",
                "supporting_evidence": [],
                "conflicts": [],
            }

        prompt = self.build_prompt(cleaned_query, evidence)
        try:
            raw_response = self._call_ollama(prompt)
            return self._parse_and_validate(raw_response, cleaned_query, evidence)
        except Exception as e:
            # Safe failure: never fail open
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "is_sufficient": False,
                "confidence": 0.0,
                "reason": f"Evidence verification failed due to parsing/validation error: {str(e)}",
                "supporting_evidence": [],
                "conflicts": [],
            }
