import json
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
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.retrieval_agent import RetrievalAgent


def test_invalid_plan_and_empty_chunks(retrieval_agent: RetrievalAgent):
    print("--- Running Test 5 & 6: Validation & Edge Cases ---")

    # TEST 5: Invalid query plan missing search_query
    invalid_plan = {
        "intent": "annual leave",
        "keywords": ["annual leave"],
        "filters": {},
    }
    try:
        retrieval_agent.retrieve(invalid_plan, chunks=[{"chunk_id": "c1", "text": "hello"}])
        assert False, "Failed to reject invalid query plan missing search_query"
    except ValueError as e:
        print(f"Test 5 (Invalid query plan rejected successfully): {e}")

    # TEST 6: Empty chunks input
    valid_plan = {
        "original_query": "Test query",
        "intent": "test",
        "search_query": "test query",
        "keywords": ["test"],
        "filters": {},
    }
    res_empty = retrieval_agent.retrieve(valid_plan, chunks=[])
    assert res_empty == [], f"Expected empty list for empty chunks, got {res_empty}"
    print("Test 6 (Empty chunks handled safely): Passed\n")


def test_end_to_end_retrieval(
    query_agent: QueryUnderstandingAgent,
    retrieval_agent: RetrievalAgent,
    chunks: list[dict],
):
    # -------------------------------------------------------------
    # TEST 1: Annual Leave
    # -------------------------------------------------------------
    q1 = "How many days of annual leave do employees receive?"
    print("==================================================")
    print(f"--- TEST 1: {q1} ---")
    plan1 = query_agent.analyze(q1)
    print(f"Generated Query Plan:\n{json.dumps(plan1, indent=2)}")

    evidence1 = retrieval_agent.retrieve(plan1, chunks=chunks, top_k=3)
    print(f"\nRetrieved Evidence ({len(evidence1)} items):")
    for rank, e in enumerate(evidence1, start=1):
        print(f"\nRank {rank}:")
        print(f"  Chunk ID: {e['chunk_id']}")
        print(f"  Page: {e['page_number']}")
        print(f"  Rerank Score: {e['rerank_score']:.4f}")
        print(f"  Hybrid Score: {e.get('hybrid_score', 0):.4f}")
        print(f"  Text: {e['text'][:150]}...")

    # Assertions
    assert len(evidence1) > 0, "No evidence returned for annual leave"
    # Verify Page 2 is present
    retrieved_pages_1 = [e["page_number"] for e in evidence1]
    assert 2 in retrieved_pages_1, f"Page 2 not found in retrieved evidence: {retrieved_pages_1}"
    # Verify fields
    for e in evidence1:
        assert "chunk_id" in e and "page_number" in e and "text" in e
        assert "rerank_score" in e and "hybrid_score" in e
    # Verify deduplication
    cids_1 = [e["chunk_id"] for e in evidence1]
    assert len(cids_1) == len(set(cids_1)), "Duplicate chunk IDs found in evidence"
    # Verify sorted descending by rerank_score
    scores_1 = [e["rerank_score"] for e in evidence1]
    assert scores_1 == sorted(scores_1, reverse=True), "Evidence not sorted descending by rerank_score"
    print("\nTEST 1 Passed.")

    # -------------------------------------------------------------
    # TEST 2: Remote Work
    # -------------------------------------------------------------
    q2 = "Can employees work from home three days a week?"
    print("\n==================================================")
    print(f"--- TEST 2: {q2} ---")
    plan2 = query_agent.analyze(q2)
    print(f"Generated Query Plan:\n{json.dumps(plan2, indent=2)}")

    evidence2 = retrieval_agent.retrieve(plan2, chunks=chunks, top_k=3)
    print(f"\nRetrieved Evidence ({len(evidence2)} items):")
    for rank, e in enumerate(evidence2, start=1):
        print(f"\nRank {rank}:")
        print(f"  Chunk ID: {e['chunk_id']} | Page: {e['page_number']}")
        print(f"  Rerank Score: {e['rerank_score']:.4f}")
        print(f"  Text: {e['text'][:150]}...")

    # Remote work policy is on Page 3
    found_remote = any(
        "remote" in e["text"].lower() or "work from home" in e["text"].lower()
        for e in evidence2
    )
    assert found_remote, "Remote work content not found in retrieved evidence"
    print("\nTEST 2 Passed.")

    # -------------------------------------------------------------
    # TEST 3: Password Policy
    # -------------------------------------------------------------
    q3 = "What is the password policy?"
    print("\n==================================================")
    print(f"--- TEST 3: {q3} ---")
    plan3 = query_agent.analyze(q3)
    print(f"Generated Query Plan:\n{json.dumps(plan3, indent=2)}")

    evidence3 = retrieval_agent.retrieve(plan3, chunks=chunks, top_k=3)
    print(f"\nRetrieved Evidence ({len(evidence3)} items):")
    for rank, e in enumerate(evidence3, start=1):
        print(f"\nRank {rank}:")
        print(f"  Chunk ID: {e['chunk_id']} | Page: {e['page_number']}")
        print(f"  Rerank Score: {e['rerank_score']:.4f}")
        print(f"  Text: {e['text'][:150]}...")

    # Password policy is on Page 5
    found_password = any("password" in e["text"].lower() for e in evidence3)
    assert found_password, "Password policy content not found in retrieved evidence"
    print("\nTEST 3 Passed.")

    # -------------------------------------------------------------
    # TEST 4: Unsupported Topic (Parental Leave)
    # -------------------------------------------------------------
    q4 = "What is the parental leave duration?"
    print("\n==================================================")
    print(f"--- TEST 4: {q4} ---")
    plan4 = query_agent.analyze(q4)
    print(f"Generated Query Plan:\n{json.dumps(plan4, indent=2)}")

    evidence4 = retrieval_agent.retrieve(plan4, chunks=chunks, top_k=3)
    print(f"\nRetrieved Evidence ({len(evidence4)} items):")
    for rank, e in enumerate(evidence4, start=1):
        print(f"\nRank {rank}:")
        print(f"  Chunk ID: {e['chunk_id']} | Page: {e['page_number']}")
        print(f"  Rerank Score: {e['rerank_score']:.4f}")
        print(f"  Text: {e['text'][:150]}...")

    # Ensure retrieval agent returns evidence list without making an answering/insufficient-evidence decision
    assert isinstance(evidence4, list), "Evidence returned is not a list"
    print("\nTEST 4 Passed.")


def main():
    print("Preparing ingestion and retrieval components...")
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    )

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    embedding_model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    reranker = CrossEncoderReranker()

    query_agent = QueryUnderstandingAgent(model_name="llama3:latest")
    retrieval_agent = RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        embedding_model=embedding_model,
    )

    test_invalid_plan_and_empty_chunks(retrieval_agent)
    test_end_to_end_retrieval(query_agent, retrieval_agent, chunks)

    print("\nAll RetrievalAgent tests passed successfully!")


if __name__ == "__main__":
    main()
