import pymupdf


def extract_text_from_pdf(
    pdf_path: str,
    document_id: str = None,
    document_name: str = None,
) -> list[dict]:
    document = pymupdf.open(pdf_path)

    # If document_name is not provided, derive from filename
    if not document_name:
        import os
        document_name = os.path.basename(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        page_meta = {
            "page_number": page_number,
            "text": text,
        }
        if document_id:
            page_meta["document_id"] = document_id
        if document_name:
            page_meta["document_name"] = document_name

        pages.append(page_meta)

    document.close()

    return pages
