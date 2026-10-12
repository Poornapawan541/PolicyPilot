"""
Unified LLM client for PolicyPilot.
Supports GroqCloud hosted inference (OpenAI-compatible chat completions)
and local Ollama inference with transparent provider selection via environment variables.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


class LLMClient:
    """
    Client providing unified completion generation for Groq and Ollama.
    """

    SUPPORTED_PROVIDERS = {"groq", "ollama"}

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_api_url: Optional[str] = None,
    ):
        raw_provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        norm_provider = raw_provider.lower().strip() if isinstance(raw_provider, str) else ""

        if norm_provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{raw_provider}'. Supported providers are: {', '.join(sorted(self.SUPPORTED_PROVIDERS))}."
            )

        self.provider = norm_provider

        if self.provider == "groq":
            self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            self.groq_api_url = groq_api_url or os.getenv("GROQ_API_URL", GROQ_API_URL)
            self.ollama_url = None
        else:
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
            self.groq_api_key = None
            self.groq_api_url = None

    def generate(
        self,
        prompt: str,
        json_format: bool = False,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> str:
        """
        Generates completion using the configured provider.
        """
        if self.provider == "groq":
            return self._call_groq(prompt, json_format=json_format, temperature=temperature, timeout=timeout)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, json_format=json_format, temperature=temperature, timeout=timeout)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_groq(
        self,
        prompt: str,
        json_format: bool = False,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> str:
        if not self.groq_api_key or not self.groq_api_key.strip():
            raise RuntimeError(
                "Groq provider selected but GROQ_API_KEY environment variable is missing or empty."
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
        }

        if json_format:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.groq_api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key.strip()}",
                "User-Agent": "PolicyPilot/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Groq API returned HTTP status {response.status}.")
                raw_bytes = response.read()
                try:
                    result = json.loads(raw_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise RuntimeError("Groq API returned a malformed response that could not be decoded as JSON.")

                if not isinstance(result, dict):
                    raise RuntimeError("Groq API response payload is not a valid JSON object.")

                choices = result.get("choices")
                if not isinstance(choices, list) or len(choices) == 0:
                    raise RuntimeError("Groq API response is missing choices array.")

                first_choice = choices[0]
                if not isinstance(first_choice, dict) or "message" not in first_choice:
                    raise RuntimeError("Groq API response choice is missing message object.")

                message = first_choice.get("message")
                if not isinstance(message, dict) or "content" not in message:
                    raise RuntimeError("Groq API response message is missing content.")

                content = message.get("content")
                if content is None or not str(content).strip():
                    raise RuntimeError("Groq API returned an empty completion content.")

                return str(content).strip()
        except urllib.error.HTTPError as e:
            # Report HTTP status code and a sanitized status message without leaking raw response bodies
            status_desc = e.reason if e.reason else "Request failed"
            raise RuntimeError(f"Groq API HTTP error {e.code}: {status_desc}.")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to communicate with Groq API: {e.reason if hasattr(e, 'reason') else e}.")

    def _call_ollama(
        self,
        prompt: str,
        json_format: bool = False,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if json_format:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP status {response.status}")
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to communicate with Ollama at {self.ollama_url}: {e}")
