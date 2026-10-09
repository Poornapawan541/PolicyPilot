from pathlib import Path


class CitationFormatter:
    """
    Formats structured, deterministic citations based strictly on retrieved
    evidence metadata and document names without LLM intervention.
    """

    def clean_document_name(self, document_name: str) -> str:
        """
        Converts file paths or filenames (e.g. 'PolicyPilot_Test_Enterprise_Handbook.pdf')
        into a clean, readable name ('PolicyPilot Test Enterprise Handbook').
        """
        if not document_name:
            return "Document"

        name = Path(document_name).stem
        cleaned = name.replace("_", " ").replace("-", " ").strip()
        return cleaned if cleaned else "Document"

    def format_citation(
        self,
        result: dict,
        document_name: str = "PolicyPilot Test Enterprise Handbook",
    ) -> dict | None:
        """
        Formats a single retrieved chunk into a citation dictionary.
        Returns None if page_number is missing or result is invalid.
        """
        if not isinstance(result, dict):
            return None

        page_number = result.get("page_number")
        if page_number is None:
            return None

        doc_name = self.clean_document_name(document_name)
        citation_str = f"{doc_name} — Page {page_number}"

        return {
            "chunk_id": result.get("chunk_id", ""),
            "document": doc_name,
            "page": page_number,
            "citation": citation_str,
        }

    def format_citations(
        self,
        results: list[dict],
        document_name: str = "PolicyPilot Test Enterprise Handbook",
        answer: str = "",
    ) -> list[dict]:
        """
        Extracts and deduplicates citations for multiple retrieved chunks.
        Safely returns an empty list if:
        - results is empty
        - answer indicates 'Insufficient evidence'
        """
        if not results:
            return []

        # If answer is 'Insufficient evidence', do not produce fake citations
        if answer:
            lower_ans = answer.strip().lower()
            if "insufficient evidence" in lower_ans or lower_ans.startswith("insufficient evidence"):
                return []

        citations = []
        seen_pages = set()

        for res in results:
            formatted = self.format_citation(res, document_name=document_name)
            if not formatted:
                continue

            page = formatted["page"]
            if page not in seen_pages:
                seen_pages.add(page)
                citations.append(formatted)

        return citations
