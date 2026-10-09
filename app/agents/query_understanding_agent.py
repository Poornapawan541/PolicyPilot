import json
import re
import urllib.request
import urllib.error


class QueryUnderstandingAgent:
    """
    Agent responsible for analyzing a natural language question and producing
    a structured retrieval plan (intent, search query, keywords, filters)
    without answering the question or retrieving documents.
    """

    def __init__(
        self,
        model_name: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

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
