import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based document chunk retriever using BM25Okapi.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple, clean tokenization: lowercase and alphanumeric word extraction.
        """
        if not text:
            return []
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves top_k relevant chunks using BM25 keyword matching.

        Returns list of structured chunks with their bm25_score.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Pair scores with their original chunk indices
        scored_indices = sorted(
            enumerate(doc_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for index, score in scored_indices[:top_k]:
            chunk = self.chunks[index]
            res = {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk.get("chunk_index", 1),
                "text": chunk["text"],
                "bm25_score": float(score),
            }
            if "document_id" in chunk:
                res["document_id"] = chunk["document_id"]
            if "document_name" in chunk:
                res["document_name"] = chunk["document_name"]

            results.append(res)

        return results
