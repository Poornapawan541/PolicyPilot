import os
from fastembed.rerank.cross_encoder import TextCrossEncoder


class CrossEncoderReranker:
    """
    Reranks candidate document chunks using an ONNX-powered Cross-Encoder model via FastEmbed.
    Evaluates (query, document) pairs directly to output relevance scores with low memory footprint.

    Implements process-level model caching to guarantee that multiple instantiations
    reuse the exact same underlying ONNX session without duplicating memory.
    """

    _cached_model = None
    _cached_model_name = None

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        if CrossEncoderReranker._cached_model is None or CrossEncoderReranker._cached_model_name != model_name:
            CrossEncoderReranker._cached_model = TextCrossEncoder(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=1,
            )
            CrossEncoderReranker._cached_model_name = model_name
        self.model = CrossEncoderReranker._cached_model

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

        # Build list of candidate texts for TextCrossEncoder
        valid_candidates = []
        texts = []
        for cand in unique_candidates:
            text = cand.get("text", "")
            if not text:
                continue
            texts.append(text)
            valid_candidates.append(cand)

        if not texts:
            return []

        # Predict scores using FastEmbed TextCrossEncoder
        scores = list(self.model.rerank(query, texts))

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
