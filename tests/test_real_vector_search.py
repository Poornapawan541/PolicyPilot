import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore


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

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    # 3. Generate embeddings for all chunks
    embedding_model = EmbeddingModel()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(texts)

    print(f"Embeddings generated: {len(embeddings)}")

    # 4. Create FAISS vector store
    vector_store = VectorStore(dimension=384)

    # 5. Store embeddings
    vector_store.add_embeddings(embeddings)

    print(f"Vectors stored in FAISS: {vector_store.count()}")

    # 6. Create a real user question
    question = "How many days of annual leave are employees entitled to?"

    print(f"\nQuestion: {question}")

    # 7. Embed the question
    query_embedding = embedding_model.embed_text(question)

    # 8. Search FAISS
    scores, indices = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print("\n--- Search Results ---")

    for rank, (score, index) in enumerate(zip(scores, indices), start=1):
        if index < 0:
            continue

        chunk = chunks[int(index)]

        print(f"\nResult {rank}")
        print(f"Score: {float(score):.4f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Text: {chunk['text'][:300]}")

    print("\nReal vector search test completed successfully.")


if __name__ == "__main__":
    main()
