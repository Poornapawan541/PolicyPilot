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

    # 1. Extract pages
    pages = extract_text_from_pdf(str(pdf_path))

    # 2. Create chunks
    chunks = chunk_pages(pages)

    # 3. Generate embeddings
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(texts)

    # 4. Create VectorStore and add embeddings
    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    # 5. Create SemanticRetriever
    retriever = SemanticRetriever(vector_store)

    # 6. Embed query
    question = "How many days of annual leave do employees receive?"
    print(f"Question: {question}\n")

    query_embedding = embedding_model.embed_text(question)

    # 7. Retrieve top-k chunks
    results = retriever.retrieve(
        query_embedding=query_embedding,
        chunks=chunks,
        top_k=3,
    )

    print(f"Retrieved results: {len(results)}")

    # 8. Validate results exist
    if not results:
        print("Error: No results retrieved.")
        return

    # Print results and validate schema
    required_keys = {"chunk_id", "page_number", "text", "similarity_score"}
    for rank, result in enumerate(results, start=1):
        missing = required_keys - set(result.keys())
        if missing:
            print(f"Error: Result {rank} missing fields: {missing}")
            return

        print(f"\n--- Result {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {result['page_number']}")
        print(f"Score: {result['similarity_score']:.4f}")
        print(f"Text: {result['text'][:300]}")

    # 9. Validate strongest result is page 2
    if results[0]["page_number"] != 2:
        print(
            f"Error: Expected top result on page 2, but got page {results[0]['page_number']}."
        )
        return

    print("\nSemantic retriever test passed.")


if __name__ == "__main__":
    main()
