from app.retrieval.retriever import SemanticRetriever
from app.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Combines dense semantic retrieval (FAISS) and sparse keyword retrieval (BM25)
    using min-max score normalization and weighted linear combination.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """
        Min-max normalizes a list of scores to [0.0, 1.0].
        Handles edge cases such as empty lists or max == min safely without division by zero.
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # If all scores are equal (e.g. all 0.0 or all identical), assign 1.0 if positive else 0.0
            return [1.0 if max_score > 0 else 0.0 for _ in scores]

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        chunks: list[dict],
        top_k: int = 5,
        candidate_k: int = 5,
    ) -> list[dict]:
        """
        Executes both FAISS and BM25 retrievals, deduplicates chunks by chunk_id,
        applies min-max score normalization, and ranks by weighted hybrid score.
        """
        # 1. Candidate retrieval from both sources
        semantic_candidates = self.semantic_retriever.retrieve(
            query_embedding=query_embedding,
            chunks=chunks,
            top_k=candidate_k,
        )

        bm25_candidates = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # 2. Extract and normalize semantic scores
        semantic_raw_scores = [c["similarity_score"] for c in semantic_candidates]
        norm_semantic_scores = self._normalize_scores(semantic_raw_scores)

        # 3. Extract and normalize BM25 scores
        bm25_raw_scores = [c["bm25_score"] for c in bm25_candidates]
        norm_bm25_scores = self._normalize_scores(bm25_raw_scores)

        # 4. Map candidate records by chunk_id
        merged_candidates = {}

        for cand, norm_score in zip(semantic_candidates, norm_semantic_scores):
            cid = cand["chunk_id"]
            merged_entry = {
                "chunk_id": cid,
                "page_number": cand["page_number"],
                "text": cand["text"],
                "semantic_score": cand["similarity_score"],
                "bm25_score": 0.0,
                "normalized_semantic_score": norm_score,
                "normalized_bm25_score": 0.0,
            }
            if "document_id" in cand:
                merged_entry["document_id"] = cand["document_id"]
            if "document_name" in cand:
                merged_entry["document_name"] = cand["document_name"]
            merged_candidates[cid] = merged_entry

        for cand, norm_score in zip(bm25_candidates, norm_bm25_scores):
            cid = cand["chunk_id"]
            if cid in merged_candidates:
                merged_candidates[cid]["bm25_score"] = cand["bm25_score"]
                merged_candidates[cid]["normalized_bm25_score"] = norm_score
                if "document_id" in cand and "document_id" not in merged_candidates[cid]:
                    merged_candidates[cid]["document_id"] = cand["document_id"]
                if "document_name" in cand and "document_name" not in merged_candidates[cid]:
                    merged_candidates[cid]["document_name"] = cand["document_name"]
            else:
                merged_entry = {
                    "chunk_id": cid,
                    "page_number": cand["page_number"],
                    "text": cand["text"],
                    "semantic_score": 0.0,
                    "bm25_score": cand["bm25_score"],
                    "normalized_semantic_score": 0.0,
                    "normalized_bm25_score": norm_score,
                }
                if "document_id" in cand:
                    merged_entry["document_id"] = cand["document_id"]
                if "document_name" in cand:
                    merged_entry["document_name"] = cand["document_name"]
                merged_candidates[cid] = merged_entry

        # 5. Compute combined hybrid scores
        results = []
        for cand in merged_candidates.values():
            h_score = (
                self.semantic_weight * cand["normalized_semantic_score"]
                + self.bm25_weight * cand["normalized_bm25_score"]
            )
            cand["hybrid_score"] = round(h_score, 4)
            # Add similarity_score alias for backward compatibility with RAG QA/citations
            cand["similarity_score"] = cand["hybrid_score"]
            results.append(cand)

        # 6. Sort descending by hybrid_score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results[:top_k]
