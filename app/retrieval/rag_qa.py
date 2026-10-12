from typing import Optional
from app.retrieval.citations import CitationFormatter
from app.core.llm import LLMClient


class RAGQuestionAnswerer:
    """
    Generates grounded answers to user questions based strictly on retrieved
    document evidence using LLM inference (Groq or local Ollama), and formats deterministic citations.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        citation_formatter: Optional[CitationFormatter] = None,
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
        Sends the grounded prompt to the LLM (Groq or Ollama) and returns the generated answer.
        Maintains backward compatibility with Step 7.
        """
        prompt = self.build_prompt(question, retrieved_chunks)
        return self.llm_client.generate(prompt, json_format=False, temperature=0.0)

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
