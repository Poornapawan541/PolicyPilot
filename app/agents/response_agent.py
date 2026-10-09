import json
import re
import urllib.request
import urllib.error
from typing import Any, Optional


class ResponseAgent:
    """
    Response / Citation Agent (Step 15).

    The final response-generation layer in PolicyPilot.
    Takes grounded conclusions from ReasoningAgent and verified evidence from
    RetrievalAgent / VerificationAgent, produces a clean, user-facing answer,
    and appends deterministic citations.

    Never performs retrieval, verification, independent reasoning, or citation invention.
    """

    ALLOWED_STATUSES = {"grounded", "insufficient", "conflicting"}

    INSUFFICIENT_ANSWER = "Insufficient evidence in the provided documents to answer this question."
    CONFLICTING_ANSWER = "The provided documents contain conflicting information, so a reliable answer cannot be determined."

    def __init__(
        self,
        model_name: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def build_prompt(
        self,
        query: str,
        conclusion: str,
        verified_evidence: list[dict],
    ) -> str:
        """
        Builds the prompt instructing the LLM to format the grounded conclusion cleanly.
        The LLM is explicitly forbidden from introducing new facts or inventing citations.
        """
        snippets = []
        for item in verified_evidence:
            cid = item.get("chunk_id", "unknown")
            page = item.get("page_number", "unknown")
            doc = item.get("document_name") or item.get("document_id") or ""
            doc_str = f" | Document: {doc}" if doc else ""
            text = item.get("text", "").strip()
            snippets.append(f"[Chunk: {cid} | Page: {page}{doc_str}]\n{text}")

        formatted_evidence = "\n\n".join(snippets)

        prompt = f"""You are a response formatting agent for an enterprise document QA system.
Your ONLY role is to deliver a concise, natural, professional user-facing answer to the user's question.

CRITICAL RULES:
1. Answer the user's question using ONLY the supplied verified conclusion and evidence text.
2. Do NOT add new facts, unstated policies, or assumptions.
3. Do NOT introduce outside knowledge.
4. Do NOT modify numbers, dates, rules, or conditions (e.g., preserve exact numbers like 24 days, 3 days, 12 characters).
5. Do NOT create or append citations (such as [Page X] or (Page X)) - citations will be appended by the system.
6. Return ONLY the direct user-facing answer text with no preamble, conversational filler, or markdown code blocks.

User Question:
{query}

Verified Conclusion:
{conclusion}

Verified Evidence:
{formatted_evidence}

Answer:"""
        return prompt

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
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
        reasoning_result: dict,
        evidence: list[dict],
    ) -> None:
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("original_query must be a non-empty string.")

        if not isinstance(reasoning_result, dict):
            raise ValueError("reasoning_result must be a dictionary.")

        if "status" not in reasoning_result:
            raise ValueError("reasoning_result must contain 'status'.")

        status = reasoning_result["status"]
        if status not in self.ALLOWED_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {self.ALLOWED_STATUSES}")

        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list.")

        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                raise ValueError(f"Evidence item at index {idx} must be a dictionary.")
            if "chunk_id" not in item:
                raise ValueError(f"Evidence item at index {idx} missing 'chunk_id'.")
            if "page_number" not in item:
                raise ValueError(f"Evidence item at index {idx} missing 'page_number'.")

    def _build_deterministic_citations(
        self,
        evidence_used: list[dict],
        evidence_by_id: dict[str, dict],
    ) -> list[dict]:
        """
        Builds deterministic citations for each valid evidence chunk used.
        Supports multi-document citations: e.g. [Document Name, Page X] or [Page X].
        """
        citations = []
        seen_tags = set()

        # Check if multiple documents are present across all supplied evidence
        distinct_docs = {
            item.get("document_name") or item.get("document_id")
            for item in evidence_by_id.values()
            if (item.get("document_name") or item.get("document_id"))
        }

        for item in evidence_used:
            chunk_id = item.get("chunk_id")
            source_item = evidence_by_id[chunk_id]
            page_num = source_item.get("page_number", item.get("page_number"))
            try:
                page_int = int(page_num)
            except (ValueError, TypeError):
                page_int = page_num

            doc_name = (
                source_item.get("document_name")
                or source_item.get("document_id")
                or item.get("document_name")
                or item.get("document_id")
            )

            # Format cleanly: strip .pdf extension for human-friendly citations if desired
            if doc_name and len(distinct_docs) > 1:
                clean_title = doc_name.replace(".pdf", "").replace("_", " ")
                citation_str = f"[{clean_title}, Page {page_int}]"
            else:
                citation_str = f"[Page {page_int}]"

            cit_meta = {
                "chunk_id": chunk_id,
                "page_number": page_int,
                "citation": citation_str,
            }
            if doc_name:
                cit_meta["document_name"] = doc_name
            if source_item.get("document_id"):
                cit_meta["document_id"] = source_item.get("document_id")

            citations.append(cit_meta)
            seen_tags.add(citation_str)

        return citations

    def respond(
        self,
        original_query: str,
        reasoning_result: dict,
        evidence: list[dict],
    ) -> dict:
        """
        Produces a grounded, user-facing response with deterministic citations.

        Input:
            original_query: str
            reasoning_result: dict (from ReasoningAgent)
            evidence: list[dict] (from RetrievalAgent)

        Output:
            {
                "original_query": str,
                "status": "grounded" | "insufficient" | "conflicting",
                "answer": str,
                "citations": list[dict]
            }
        """
        self._validate_inputs(original_query, reasoning_result, evidence)
        cleaned_query = original_query.strip()
        status = reasoning_result["status"]

        # Gate 1: Insufficient evidence status
        if status == "insufficient":
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "answer": self.INSUFFICIENT_ANSWER,
                "citations": [],
            }

        # Gate 2: Conflicting evidence status
        if status == "conflicting":
            conflicts = reasoning_result.get("conflicts", [])
            if conflicts and isinstance(conflicts, list):
                # Build explanation identifying the conflicting sources
                first_conflict = conflicts[0]
                topic = first_conflict.get("topic", "the requested topic")
                claims = first_conflict.get("claims", [])
                source_descriptions = []
                for cl in claims:
                    doc = cl.get("document_name") or cl.get("document_id") or "Document"
                    page = cl.get("page_number", "")
                    clean_doc = doc.replace(".pdf", "").replace("_", " ") if doc else "Document"
                    loc = f"{clean_doc} (Page {page})" if page else clean_doc
                    claim_text = cl.get("claim", "").strip()
                    if claim_text:
                        source_descriptions.append(f"{loc} states: \"{claim_text}\"")
                    else:
                        source_descriptions.append(loc)

                if source_descriptions:
                    desc_str = "; while ".join(source_descriptions)
                    answer = f"The provided documents contain conflicting information regarding {topic}: {desc_str}. A reliable answer cannot be determined."
                else:
                    answer = self.CONFLICTING_ANSWER
            else:
                answer = self.CONFLICTING_ANSWER

            return {
                "original_query": cleaned_query,
                "status": "conflicting",
                "answer": answer,
                "citations": [],
            }

        # Gate 3: Empty evidence list provided
        if not evidence:
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "answer": self.INSUFFICIENT_ANSWER,
                "citations": [],
            }

        conclusion = reasoning_result.get("conclusion")
        if not conclusion or not isinstance(conclusion, str) or not conclusion.strip():
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "answer": self.INSUFFICIENT_ANSWER,
                "citations": [],
            }

        evidence_used = reasoning_result.get("evidence_used", [])
        if not isinstance(evidence_used, list) or not evidence_used:
            return {
                "original_query": cleaned_query,
                "status": "insufficient",
                "answer": self.INSUFFICIENT_ANSWER,
                "citations": [],
            }

        # Validate that every referenced chunk exists in supplied evidence
        evidence_by_id = {item["chunk_id"]: item for item in evidence}
        verified_evidence_list = []

        for item in evidence_used:
            if not isinstance(item, dict) or "chunk_id" not in item:
                return {
                    "original_query": cleaned_query,
                    "status": "insufficient",
                    "answer": self.INSUFFICIENT_ANSWER,
                    "citations": [],
                }
            chunk_id = item["chunk_id"]
            if chunk_id not in evidence_by_id:
                # Unknown/hallucinated chunk ID -> Fail closed safely
                return {
                    "original_query": cleaned_query,
                    "status": "insufficient",
                    "answer": self.INSUFFICIENT_ANSWER,
                    "citations": [],
                }
            verified_evidence_list.append(evidence_by_id[chunk_id])

        # Generate deterministic citations
        citations = self._build_deterministic_citations(evidence_used, evidence_by_id)

        # Call LLM to format the response
        prompt = self.build_prompt(cleaned_query, conclusion.strip(), verified_evidence_list)
        try:
            raw_answer = self._call_ollama(prompt)
            if not raw_answer or not raw_answer.strip():
                # Fallback to conclusion if LLM returns empty output
                raw_answer = conclusion.strip()
            else:
                raw_answer = raw_answer.strip()
                # Clean up if model erroneously appended quotes or code block
                if raw_answer.startswith("```"):
                    raw_answer = re.sub(r"^```(?:json)?\s*", "", raw_answer)
                    raw_answer = re.sub(r"\s*```$", "", raw_answer).strip()
                # Strip out any citations the model might have tried to generate in text
                raw_answer = re.sub(r"\[(?:[^\]]+,\s*)?Page\s*\d+\]", "", raw_answer, flags=re.IGNORECASE).strip()

            # Append deterministic citations string
            unique_page_citations = []
            seen = set()
            for c in citations:
                cit_tag = c["citation"]
                if cit_tag not in seen:
                    unique_page_citations.append(cit_tag)
                    seen.add(cit_tag)

            citation_suffix = " " + " ".join(unique_page_citations) if unique_page_citations else ""
            final_answer = f"{raw_answer}{citation_suffix}"

            return {
                "original_query": cleaned_query,
                "status": "grounded",
                "answer": final_answer,
                "citations": citations,
            }

        except Exception:
            # Safe fallback: never fail open, use grounded conclusion directly with citations
            unique_page_citations = []
            seen = set()
            for c in citations:
                cit_tag = c["citation"]
                if cit_tag not in seen:
                    unique_page_citations.append(cit_tag)
                    seen.add(cit_tag)
            citation_suffix = " " + " ".join(unique_page_citations) if unique_page_citations else ""
            final_answer = f"{conclusion.strip()}{citation_suffix}"

            return {
                "original_query": cleaned_query,
                "status": "grounded",
                "answer": final_answer,
                "citations": citations,
            }
