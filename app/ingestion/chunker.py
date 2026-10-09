def chunk_pages(
    pages: list[dict],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """
    Splits page-extracted text into overlapping word-based chunks while preserving
    page numbers and metadata for citation.

    Args:
        pages: List of dictionaries containing 'page_number' and 'text'.
        chunk_size: Approximate maximum number of words per chunk.
        overlap: Number of words repeated between consecutive chunks of the same page.

    Returns:
        List of chunk dictionaries with metadata (chunk_id, page_number, chunk_index, text).
    """
    chunks = []

    # Step determines how many new words we advance after each chunk
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for page in pages:
        page_number = page.get("page_number")
        text = page.get("text", "").strip()
        doc_id = page.get("document_id")
        doc_name = page.get("document_name")

        # Skip pages with no text content
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        chunk_index = 1
        for start_idx in range(0, len(words), step):
            chunk_words = words[start_idx : start_idx + chunk_size]
            chunk_text = " ".join(chunk_words)

            if doc_id:
                cid = f"{doc_id}-page-{page_number}-chunk-{chunk_index}"
            else:
                cid = f"page-{page_number}-chunk-{chunk_index}"

            chunk_meta = {
                "chunk_id": cid,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            }
            if doc_id:
                chunk_meta["document_id"] = doc_id
            if doc_name:
                chunk_meta["document_name"] = doc_name

            chunks.append(chunk_meta)

            chunk_index += 1

            # If this chunk reached or covered the end of the page's words, break
            if start_idx + chunk_size >= len(words):
                break

    return chunks
