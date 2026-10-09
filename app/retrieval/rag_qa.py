import json
import urllib.request
import urllib.error
from app.retrieval.citations import CitationFormatter


class RAGQuestionAnswerer:
    """
    Generates grounded answers to user questions based strictly on retrieved
    document evidence using a local Ollama LLM, and formats deterministic citations.
    """

    def __init__(
        self,
        model_name: str = "llama3:latest",
        ollama_url: str = "http://localhost:11434/api/generate",
        citation_formatter: CitationFormatter | None = None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.citation_formatter = citation_formatter or CitationFormatter()

    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        Constructs a strictly grounded prompt containing only the retrieved chunks
        and anti-hallucination instructions.
        """
        evidence_text_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = chunk.get("chunk_id", f"chunk-{i}")
            page_num = chunk.get("page_number", "Unknown")
            text = chunk.get("text", "").strip()
            evidence_text_blocks.append(
                f"[Evidence {i} | ID: {chunk_id} | Page: {page_num}]\n{text}"
            )

        combined_evidence = (
            "\n\n".join(evidence_text_blocks)
            if evidence_text_blocks
            else "No evidence provided."
        )

        prompt = f"""You are PolicyPilot, an enterprise knowledge assistant.

Answer the user's question using ONLY the provided document evidence.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the evidence does not contain enough information to answer the question, respond: "Insufficient evidence."
4. Be concise and directly answer the question.
5. Do not mention information that is not supported by the evidence.

DOCUMENT EVIDENCE:
{combined_evidence}

USER QUESTION:
{question}

ANSWER:"""
        return prompt

    def answer_question(
        self,
        question: str,
        retrieved_chunks: list[dict],
        document_name: str = "PolicyPilot Test Enterprise Handbook",
    ) -> str:
        """
        Sends the grounded prompt to the Ollama HTTP API and returns the generated answer.
        Maintains backward compatibility with Step 7.
        """
        prompt = self.build_prompt(question, retrieved_chunks)

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

    def answer_with_citations(
        self,
        question: str,
        retrieved_chunks: list[dict],
        document_name: str = "PolicyPilot Test Enterprise Handbook",
    ) -> dict:
        """
        Generates the grounded answer via LLM and builds deterministic citations
        from retrieved metadata using CitationFormatter.
        """
        answer = self.answer_question(question, retrieved_chunks, document_name=document_name)
        citations = self.citation_formatter.format_citations(
            results=retrieved_chunks,
            document_name=document_name,
            answer=answer,
        )

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
        }
