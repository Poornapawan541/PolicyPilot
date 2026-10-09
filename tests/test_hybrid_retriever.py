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


def test_normalization_logic():
    print("--- Running Normalization Logic Tests ---")
    retriever = HybridRetriever(None, None)

    # 1. Normal list
    norm_scores = retriever._normalize_scores([10.0, 20.0, 30.0])
    assert norm_scores == [0.0, 0.5, 1.0], f"Expected [0.0, 0.5, 1.0], got {norm_scores}"

    # 2. Values between 0 and 1
    assert all(0.0 <= s <= 1.0 for s in norm_scores), "Scores outside [0, 1]"

    # 3. Identical scores (division by zero prevention)
    identical_scores = retriever._normalize_scores([5.0, 5.0, 5.0])
    assert identical_scores == [1.0, 1.0, 1.0], f"Expected [1.0, 1.0, 1.0], got {identical_scores}"

    # 4. Empty list
    empty_scores = retriever._normalize_scores([])
    assert empty_scores == [], "Expected empty list"

    print("Normalization logic tests passed.\n")


def test_hybrid_flow():
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

    # Dense setup
    embedding_model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)
    semantic_retriever = SemanticRetriever(vector_store)

    # Sparse setup
    bm25_retriever = BM25Retriever(chunks)

    # Hybrid setup
    hybrid = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        semantic_weight=0.5,
        bm25_weight=0.5,
    )

    # -------------------------------------------------------------
    # Query 1: Annual Leave
    # -------------------------------------------------------------
    query_1 = "How many days of annual leave do employees receive?"
    print(f"--- Query 1: {query_1} ---")
    query_1_emb = embedding_model.embed_text(query_1)

    results_1 = hybrid.retrieve(
        query=query_1,
        query_embedding=query_1_emb,
        chunks=chunks,
        top_k=3,
        candidate_k=5,
    )

    print(f"Retrieved {len(results_1)} hybrid results:")
    for rank, r in enumerate(results_1, start=1):
        print(f"\nResult {rank}:")
        print(f"  Chunk ID: {r['chunk_id']}")
        print(f"  Page: {r['page_number']}")
        print(f"  Semantic Score: {r['semantic_score']:.4f} (Norm: {r['normalized_semantic_score']:.4f})")
        print(f"  BM25 Score: {r['bm25_score']:.4f} (Norm: {r['normalized_bm25_score']:.4f})")
        print(f"  Hybrid Score: {r['hybrid_score']:.4f}")
        print(f"  Text: {r['text'][:150]}...")

    # Validations for Query 1
    assert len(results_1) > 0, "No results returned"
    required_keys = {
        "chunk_id",
        "page_number",
        "text",
        "semantic_score",
        "bm25_score",
        "hybrid_score",
    }
    for r in results_1:
        assert required_keys.issubset(r.keys()), f"Missing keys in result: {r.keys()}"

    # Deduplication check
    chunk_ids = [r["chunk_id"] for r in results_1]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in results"

    # Sorted order check
    scores = [r["hybrid_score"] for r in results_1]
    assert scores == sorted(scores, reverse=True), "Results not sorted by hybrid_score descending"

    # Top result page check
    assert results_1[0]["page_number"] == 2, f"Expected top result page 2, got {results_1[0]['page_number']}"
    print("\nQuery 1 assertions passed.")

    # -------------------------------------------------------------
    # Query 2: Exact Keyword Benefit (Password Policy)
    # -------------------------------------------------------------
    query_2 = "What is the password policy?"
    print(f"\n--- Query 2 (Exact Keyword Benefit): {query_2} ---")
    query_2_emb = embedding_model.embed_text(query_2)

    results_2 = hybrid.retrieve(
        query=query_2,
        query_embedding=query_2_emb,
        chunks=chunks,
        top_k=3,
        candidate_k=5,
    )

    for rank, r in enumerate(results_2, start=1):
        print(f"\nResult {rank}:")
        print(f"  Chunk ID: {r['chunk_id']} | Page: {r['page_number']}")
        print(f"  BM25 Score: {r['bm25_score']:.4f} | Hybrid Score: {r['hybrid_score']:.4f}")
        print(f"  Text: {r['text'][:150]}...")

    # The password policy chunk must be retrieved and scored highly by BM25
    password_chunks = [r for r in results_2 if "password" in r["text"].lower()]
    assert len(password_chunks) > 0, "Password chunk not found in top hybrid results"
    assert password_chunks[0]["bm25_score"] > 0, "BM25 score for password chunk was 0"
    print("\nQuery 2 (Keyword Benefit) assertions passed.")

    print("\nAll Hybrid Retriever tests passed successfully.")


def main():
    test_normalization_logic()
    test_hybrid_flow()


if __name__ == "__main__":
    main()
