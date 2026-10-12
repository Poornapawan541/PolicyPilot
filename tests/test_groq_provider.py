"""
Unit tests for Groq and Ollama provider integration via LLMClient across PolicyPilot agents.

Tests verify:
1. Default provider fallback to Ollama when LLM_PROVIDER is unset or set to 'ollama'.
2. Groq provider selection when LLM_PROVIDER='groq'.
3. Missing GROQ_API_KEY error handling without leaking secrets or crashing.
4. Correct Groq request construction (chat completions payload, Bearer token header, messages format, model selection).
5. Robust response parsing from Groq choices[0].message.content.
6. Integration across QueryUnderstandingAgent, EvidenceVerificationAgent, ReasoningAgent, ResponseAgent, and RAGQuestionAnswerer.
"""

import json
from unittest.mock import patch, MagicMock
import urllib.error

from app.core.llm import LLMClient, DEFAULT_GROQ_MODEL
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent
from app.retrieval.rag_qa import RAGQuestionAnswerer


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient()
    assert client.provider == "ollama"
    assert client.model_name == "llama3:latest"
    assert "11434" in client.ollama_url


def test_groq_provider_selection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret_12345")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    client = LLMClient()
    assert client.provider == "groq"
    assert client.model_name == DEFAULT_GROQ_MODEL
    assert client.groq_api_key == "gsk_test_secret_12345"
    assert "api.groq.com" in client.groq_api_url


def test_missing_groq_api_key_raises_informative_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    client = LLMClient()
    try:
        client.generate("Hello world")
        assert False, "Should raise RuntimeError when GROQ_API_KEY is missing"
    except RuntimeError as e:
        err = str(e)
        assert "GROQ_API_KEY" in err
        assert "missing or empty" in err
        # Ensure no accidental secret token pattern
        assert "gsk_" not in err


def test_groq_request_payload_and_response_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")

    mock_response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "original_query": "What is the policy?",
                        "intent": "policy check",
                        "search_query": "policy check enterprise",
                        "keywords": ["policy", "check"],
                        "filters": {},
                    }),
                },
                "finish_reason": "stop",
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        agent = QueryUnderstandingAgent()
        result = agent.analyze("What is the policy?")

        # Check call arguments
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer gsk_mock_valid_key"
        assert req.headers["Content-type"] == "application/json"

        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "llama-3.3-70b-versatile"
        assert body["messages"][0]["role"] == "user"
        assert body["response_format"] == {"type": "json_object"}

        # Check parsed result
        assert result["intent"] == "policy check"
        assert result["search_query"] == "policy check enterprise"


def test_evidence_verification_agent_groq_execution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    mock_groq_resp = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "original_query": "How many days leave?",
                        "status": "sufficient",
                        "is_sufficient": True,
                        "confidence": 0.95,
                        "reason": "Directly supported by page 2.",
                        "supporting_evidence": [
                            {"chunk_id": "c1", "page_number": 2}
                        ],
                        "conflicts": [],
                    })
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_groq_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        agent = EvidenceVerificationAgent()
        evidence = [{"chunk_id": "c1", "page_number": 2, "text": "Employees receive 24 days annual leave."}]
        res = agent.verify("How many days leave?", evidence)
        assert res["status"] == "sufficient"
        assert res["is_sufficient"] is True


def test_reasoning_and_response_agents_groq_execution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    # Reasoning agent response mock
    mock_reason_resp = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "original_query": "How many days leave?",
                        "status": "grounded",
                        "conclusion": "Employees receive 24 days annual leave per year.",
                        "reason": "Handbook states 24 days.",
                        "evidence_used": [{"chunk_id": "c1", "page_number": 2}],
                    })
                }
            }
        ]
    }

    mock_resp1 = MagicMock()
    mock_resp1.status = 200
    mock_resp1.read.return_value = json.dumps(mock_reason_resp).encode("utf-8")
    mock_resp1.__enter__.return_value = mock_resp1

    with patch("urllib.request.urlopen", return_value=mock_resp1):
        reason_agent = ReasoningAgent()
        ver_result = {"status": "sufficient", "is_sufficient": True}
        evidence = [{"chunk_id": "c1", "page_number": 2, "text": "24 days annual leave."}]
        reas = reason_agent.reason("How many days leave?", ver_result, evidence)
        assert reas["status"] == "grounded"
        assert "24 days" in reas["conclusion"]

    # Response agent response mock
    mock_resp_content = {
        "choices": [
            {
                "message": {
                    "content": "Employees receive 24 days of annual leave per year. [Page 2]"
                }
            }
        ]
    }
    mock_resp2 = MagicMock()
    mock_resp2.status = 200
    mock_resp2.read.return_value = json.dumps(mock_resp_content).encode("utf-8")
    mock_resp2.__enter__.return_value = mock_resp2

    with patch("urllib.request.urlopen", return_value=mock_resp2):
        resp_agent = ResponseAgent()
        resp = resp_agent.respond("How many days leave?", reas, evidence)
        assert resp["status"] == "grounded"
        assert "24 days" in resp["answer"]


def test_rag_qa_groq_execution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    mock_rag_resp = {
        "choices": [
            {
                "message": {
                    "content": "Employees get 24 days annual leave."
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(mock_rag_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        rag = RAGQuestionAnswerer()
        chunks = [{"chunk_id": "c1", "page_number": 2, "text": "Employees get 24 days annual leave."}]
        ans = rag.answer_question("How many days annual leave?", chunks)
        assert ans == "Employees get 24 days annual leave."


def test_unsupported_provider_raises_value_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    try:
        LLMClient()
        assert False, "Should raise ValueError for unsupported provider"
    except ValueError as e:
        err = str(e)
        assert "Unsupported LLM_PROVIDER 'invalid_provider'" in err
        assert "groq" in err
        assert "ollama" in err


def test_groq_http_error_sanitization(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    http_err = urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        client = LLMClient()
        try:
            client.generate("hello")
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            err = str(e)
            assert "401" in err
            assert "Unauthorized" in err
            # Must not leak key
            assert "gsk_mock_key" not in err


def test_groq_malformed_json_response(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        client = LLMClient()
        try:
            client.generate("hello")
            assert False, "Should raise RuntimeError for malformed JSON"
        except RuntimeError as e:
            err = str(e)
            assert "malformed response" in err or "JSON" in err


def test_groq_missing_choices_or_message(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    # Missing choices
    mock_resp1 = MagicMock()
    mock_resp1.status = 200
    mock_resp1.read.return_value = json.dumps({"id": "123"}).encode("utf-8")
    mock_resp1.__enter__.return_value = mock_resp1

    with patch("urllib.request.urlopen", return_value=mock_resp1):
        client = LLMClient()
        try:
            client.generate("hello")
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            assert "missing choices" in str(e)

    # Missing message
    mock_resp2 = MagicMock()
    mock_resp2.status = 200
    mock_resp2.read.return_value = json.dumps({"choices": [{}]}).encode("utf-8")
    mock_resp2.__enter__.return_value = mock_resp2

    with patch("urllib.request.urlopen", return_value=mock_resp2):
        client = LLMClient()
        try:
            client.generate("hello")
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            assert "missing message" in str(e)


def test_groq_empty_content_raises_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "   "}}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        client = LLMClient()
        try:
            client.generate("hello")
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            assert "empty completion content" in str(e)
