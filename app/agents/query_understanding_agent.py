import json
import re
from typing import Optional
from app.core.llm import LLMClient


class QueryUnderstandingAgent:
    """
    Agent responsible for analyzing a natural language question and producing
    a structured retrieval plan (intent, search query, keywords, filters)
    without answering the question or retrieving documents.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model_name=model_name,
            ollama_url=ollama_url,
            groq_api_key=groq_api_key,
        )
        self.model_name = self.llm_client.model_name
        self.ollama_url = self.llm_client.ollama_url
        self.provider = self.llm_client.provider

    def build_prompt(self, query: str) -> str:
        prompt = f"""You are a query understanding component for an enterprise document retrieval system.
You do NOT answer the user's question.
You only convert the question into a structured retrieval plan.
Do NOT invent enterprise facts or answers.
Preserve the user's original meaning.
Generate concise search terms and keywords useful for semantic and lexical search.

Output strictly valid JSON with this exact schema and no other text:
{{
  "original_query": "{query}",
  "intent": "<short description of what the user is seeking, e.g. annual leave entitlement>",
  "search_query": "<concise retrieval-friendly query without punctuation, e.g. annual leave entitlement number of days employees>",
  "keywords": ["<keyword or keyphrase 1>", "<keyword or keyphrase 2>", "<keyword 3>"],
  "filters": {{}}
}}

Rules:
1. Do NOT answer the question (e.g. do not say "24 days" or "yes").
2. "original_query" must be the exact user query provided.
3. "keywords" must be a JSON array of 3 to 8 concise, relevant terms or phrases.
4. "filters" must be a JSON object (empty dictionary {{}} unless the query specifies a strict document constraint).
5. Output ONLY raw JSON. Do not include markdown code fences, explanation, or commentary.

User Question: {query}

JSON:"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        return self.llm_client.generate(prompt, json_format=True, temperature=0.0)

    def _call_ollama(self, prompt: str) -> str:
        """Backward compatible alias for _call_llm."""
        return self._call_llm(prompt)

    def _parse_and_validate(self, raw_response: str, original_query: str) -> dict:
        """
        Parses JSON from Ollama response and validates required fields and types.
        """
        if not raw_response:
            raise ValueError("Ollama returned an empty response.")

        # Strip markdown fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Fallback: attempt regex extraction of JSON object
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
        required_fields = ["original_query", "intent", "search_query", "keywords", "filters"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in query plan: '{field}'")

        # Type checks
        if not isinstance(data["original_query"], str):
            raise ValueError("'original_query' must be a string")
        if not isinstance(data["intent"], str):
            raise ValueError("'intent' must be a string")
        if not isinstance(data["search_query"], str):
            raise ValueError("'search_query' must be a string")
        if not isinstance(data["keywords"], list):
            raise ValueError("'keywords' must be a list of strings")
        if not isinstance(data["filters"], dict):
            raise ValueError("'filters' must be a dictionary")

        # Ensure keywords are strings
        data["keywords"] = [str(k).strip() for k in data["keywords"] if str(k).strip()]
        if not data["keywords"]:
            raise ValueError("'keywords' list cannot be empty")

        # Guarantee original query is preserved
        data["original_query"] = original_query

        return {
            "original_query": data["original_query"],
            "intent": data["intent"].strip(),
            "search_query": data["search_query"].strip(),
            "keywords": data["keywords"],
            "filters": data["filters"],
        }

    def analyze(self, query: str) -> dict:
        """
        Analyzes a natural language question and returns a structured retrieval plan.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        clean_query = query.strip()
        prompt = self.build_prompt(clean_query)
        raw_output = self._call_ollama(prompt)
        return self._parse_and_validate(raw_output, clean_query)
