import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages


def test_pdf_chunks():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    pages = extract_text_from_pdf(str(pdf_path))
    print(f"Total pages extracted: {len(pages)}")

    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    print(f"Total chunks created: {len(chunks)}")

    # Verify that every chunk contains required fields
    required_fields = ["chunk_id", "page_number", "chunk_index", "text"]
    for i, chunk in enumerate(chunks):
        missing = [f for f in required_fields if f not in chunk]
        if missing:
            print(f"Error: Chunk at index {i} is missing fields: {missing}")
            return

    print("All chunks validated successfully with required fields.")

    if chunks:
        first_chunk = chunks[0]
        print("\n--- First Chunk ---")
        print(f"chunk_id: {first_chunk['chunk_id']}")
        print(f"page_number: {first_chunk['page_number']}")
        print(f"chunk_index: {first_chunk['chunk_index']}")
        print(f"text: {first_chunk['text']}")

        last_chunk = chunks[-1]
        print("\n--- Last Chunk ---")
        print(f"chunk_id: {last_chunk['chunk_id']}")
        print(f"page_number: {last_chunk['page_number']}")
        print(f"chunk_index: {last_chunk['chunk_index']}")
        print(f"text: {last_chunk['text']}")


def test_synthetic_overlap():
    print("\n--- Running Synthetic Overlap Test ---")
    # Generate 350 words: "word1 word2 ... word350"
    words = [f"word{i}" for i in range(1, 351)]
    synthetic_page = {
        "page_number": 99,
        "text": " ".join(words),
    }

    chunks = chunk_pages([synthetic_page], chunk_size=200, overlap=40)

    # 1. Verify at least 2 chunks created
    if len(chunks) < 2:
        print(f"Error: Expected at least 2 chunks, got {len(chunks)}")
        return

    # 2. Verify first chunk has 200 words
    first_chunk_words = chunks[0]["text"].split()
    if len(first_chunk_words) != 200:
        print(f"Error: Expected 200 words in first chunk, got {len(first_chunk_words)}")
        return

    # 3. Verify second chunk starts with word161
    second_chunk_words = chunks[1]["text"].split()
    expected_start_word = "word161"
    if second_chunk_words[0] != expected_start_word:
        print(
            f"Error: Expected second chunk to start with '{expected_start_word}', got '{second_chunk_words[0]}'"
        )
        return

    print("Overlap test passed: 40 words overlap between consecutive chunks.")


def main():
    test_pdf_chunks()
    test_synthetic_overlap()


if __name__ == "__main__":
    main()
