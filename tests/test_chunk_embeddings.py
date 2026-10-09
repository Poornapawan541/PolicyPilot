import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel


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

    # Step 1: Extract PDF pages
    pages = extract_text_from_pdf(str(pdf_path))
    print(f"Total pages extracted: {len(pages)}")

    # Step 2: Create chunks
    chunks = chunk_pages(pages)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Extract chunk text
    texts = [chunk["text"] for chunk in chunks]

    # Step 4: Generate embeddings
    model = EmbeddingModel()
    embeddings = model.embed_texts(texts)

    print(f"Total embeddings generated: {len(embeddings)}")

    # Step 5: Validate chunk/embedding count
    if len(embeddings) != len(chunks):
        print(
            f"Error: Number of embeddings ({len(embeddings)}) "
            f"does not match number of chunks ({len(chunks)})."
        )
        return

    # Step 6: Validate embedding dimensions
    expected_dimension = 384

    for i, embedding in enumerate(embeddings):
        if len(embedding) != expected_dimension:
            print(
                f"Error: Embedding {i} has dimension {len(embedding)}, "
                f"expected {expected_dimension}."
            )
            return

    print("\nChunk embedding test passed.")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\n--- First Chunk + Embedding Preview ---")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Page number: {chunks[0]['page_number']}")
    print(f"Text: {chunks[0]['text'][:200]}")
    print(f"First 5 embedding values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()