import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
import pytest

@pytest.fixture
def reranker():
    return CrossEncoderReranker()

def test_empty_and_sorting_logic(reranker: CrossEncoderReranker):
    print("--- Running Unit / Safety Tests ---")
    # TEST 3: Empty candidates
    res_empty = reranker.rerank("any query", [])
    assert res_empty == [], f"Expected empty list, got {res_empty}"

    res_empty_query = reranker.rerank("", [{"chunk_id": "c1", "text": "hello"}])
    assert res_empty_query == [], f"Expected empty list for empty query, got {res_empty_query}"

    # TEST 4: Result ordering logic
    dummy_candidates = [
        {"chunk_id": "c1", "page_number": 1, "text": "The quick brown fox jumps over the lazy dog."},
        {"chunk_id": "c2", "page_number": 2, "text": "Annual leave policy states employees get 24 days."},
    ]
    reranked = reranker.rerank(
        query="How many days of annual leave do employees receive?",
        candidates=dummy_candidates,
        top_k=2,
    )
    assert len(reranked) == 2, f"Expected 2 results, got {len(reranked)}"
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"], "Results not sorted descending by score"
    print("Unit / Safety tests passed.\n")


def test_real_handbook_reranking(reranker: CrossEncoderReranker):
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    )

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Ingestion
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    # Embedding & FAISS
    embedding_model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)
    semantic_retriever = SemanticRetriever(vector_store)

    # BM25
    bm25_retriever = BM25Retriever(chunks)

    # Hybrid Retriever
    hybrid = HybridRetriever(semantic_retriever, bm25_retriever)

    # -------------------------------------------------------------
    # TEST 1: Annual Leave
    # -------------------------------------------------------------
    query_1 = "How many days of annual leave do employees receive?"
    print(f"==================================================")
    print(f"--- TEST 1: {query_1} ---")
    query_1_emb = embedding_model.embed_text(query_1)

    # 1. Retrieve candidate pool using HybridRetriever (candidate_k=5)
    candidates_1 = hybrid.retrieve(
        query=query_1,
        query_embedding=query_1_emb,
        chunks=chunks,
        top_k=5,
        candidate_k=5,
    )
    print(f"Hybrid candidates retrieved: {len(candidates_1)}")

    # 2. Rerank candidates with CrossEncoder (top_k=3)
    reranked_1 = reranker.rerank(
        query=query_1,
        candidates=candidates_1,
        top_k=3,
    )

    print(f"\nReranked Results ({len(reranked_1)}):")
    for rank, r in enumerate(reranked_1, start=1):
        print(f"\nResult {rank}:")
        print(f"  Chunk ID: {r['chunk_id']}")
        print(f"  Page: {r['page_number']}")
        print(f"  Rerank Score: {r['rerank_score']:.4f}")
        print(f"  Hybrid Score: {r.get('hybrid_score', 0):.4f} (Semantic: {r.get('semantic_score', 0):.4f}, BM25: {r.get('bm25_score', 0):.4f})")
        print(f"  Text: {r['text'][:150]}...")

    # Verifications for Test 1
    assert len(reranked_1) > 0, "No reranked results returned"
    for r in reranked_1:
        assert "chunk_id" in r and "page_number" in r and "text" in r and "rerank_score" in r, "Missing fields"

    # Deduplication check
    cids_1 = [r["chunk_id"] for r in reranked_1]
    assert len(cids_1) == len(set(cids_1)), "Duplicate chunk IDs found in reranked results"

    # Descending score sort check
    scores_1 = [r["rerank_score"] for r in reranked_1]
    assert scores_1 == sorted(scores_1, reverse=True), "Results not sorted descending by rerank_score"

    # Check Page 2 is present and whether it is #1
    pages_retrieved = [r["page_number"] for r in reranked_1]
    assert 2 in pages_retrieved, f"Page 2 not found in reranked results: {pages_retrieved}"
    print(f"\nPage 2 ranked at position: {pages_retrieved.index(2) + 1} (Top score: {reranked_1[0]['rerank_score']:.4f})")
    print("TEST 1 passed.")

    # -------------------------------------------------------------
    # TEST 2: Password Policy
    # -------------------------------------------------------------
    query_2 = "What is the password policy?"
    print(f"\n==================================================")
    print(f"--- TEST 2: {query_2} ---")
    query_2_emb = embedding_model.embed_text(query_2)

    candidates_2 = hybrid.retrieve(
        query=query_2,
        query_embedding=query_2_emb,
        chunks=chunks,
        top_k=5,
        candidate_k=5,
    )

    reranked_2 = reranker.rerank(
        query=query_2,
        candidates=candidates_2,
        top_k=3,
    )

    print(f"\nReranked Results ({len(reranked_2)}):")
    for rank, r in enumerate(reranked_2, start=1):
        print(f"\nResult {rank}:")
        print(f"  Chunk ID: {r['chunk_id']}")
        print(f"  Page: {r['page_number']}")
        print(f"  Rerank Score: {r['rerank_score']:.4f}")
        print(f"  Text: {r['text'][:150]}...")

    pages_retrieved_2 = [r["page_number"] for r in reranked_2]
    assert 5 in pages_retrieved_2, f"Page 5 (Password Policy) not found in reranked results: {pages_retrieved_2}"
    print(f"\nPage 5 ranked at position: {pages_retrieved_2.index(5) + 1} (Rerank score: {reranked_2[pages_retrieved_2.index(5)]['rerank_score']:.4f})")
    print("TEST 2 passed.")


def main():
    print("Loading CrossEncoderReranker...")
    reranker = CrossEncoderReranker()
    test_empty_and_sorting_logic(reranker)
    test_real_handbook_reranking(reranker)
    print("\nAll Reranker tests completed successfully!")


if __name__ == "__main__":
    main()
