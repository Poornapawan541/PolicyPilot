from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate document chunks using a local Sentence Transformers Cross-Encoder model.
    Evaluates (query, document) pairs directly with cross-attention to output relevance scores.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Takes a query and candidate chunks, predicts cross-encoder relevance scores,
        deduplicates by chunk_id, and returns the top_k candidates sorted by rerank_score descending.

        Preserves all existing metadata and scores (chunk_id, page_number, text,
        semantic_score, bm25_score, hybrid_score).
        """
        if not candidates or not query:
            return []

        # Deduplicate candidates by chunk_id while preserving order
        unique_candidates = []
        seen_ids = set()
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cid = cand.get("chunk_id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            unique_candidates.append(cand)

        if not unique_candidates:
            return []

        # Build (query, chunk_text) pairs for CrossEncoder
        pairs = []
        valid_candidates = []
        for cand in unique_candidates:
            text = cand.get("text", "")
            if not text:
                continue
            pairs.append([query, text])
            valid_candidates.append(cand)

        if not pairs:
            return []

        # Predict scores
        scores = self.model.predict(pairs)

        # Attach rerank_score and copy
        reranked_results = []
        for cand, score in zip(valid_candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            # Update similarity_score alias to point to the rerank_score for downstream compatibility
            cand_copy["similarity_score"] = float(score)
            reranked_results.append(cand_copy)

        # Sort descending by rerank_score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:top_k]
