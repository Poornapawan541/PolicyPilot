import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval.bm25_retriever import BM25Retriever


def main():
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    )

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    # 1. Ingestion: Extract & Chunk
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    # 2. Build BM25 index
    retriever = BM25Retriever(chunks)

    # 3. Test Query
    query = "How many days of annual leave do employees receive?"
    print(f"Query: {query}\n")

    results = retriever.retrieve(query, top_k=3)

    print(f"BM25 Retrieved results: {len(results)}")
    if not results:
        print("Error: No BM25 results returned.")
        return

    for rank, res in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {res['chunk_id']}")
        print(f"Page: {res['page_number']}")
        print(f"Score: {res['bm25_score']:.4f}")
        print(f"Text: {res['text'][:200]}...")

    # Assertions
    top_result = results[0]
    assert top_result["page_number"] == 2, (
        f"Expected top BM25 result on page 2, got page {top_result['page_number']}"
    )
    assert "bm25_score" in top_result, "Missing bm25_score in result"
    assert "chunk_id" in top_result, "Missing chunk_id in result"

    print("\nBM25 retriever test passed.")


if __name__ == "__main__":
    main()
