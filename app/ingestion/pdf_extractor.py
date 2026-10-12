import os
from pathlib import Path
from typing import Optional
import pypdf


def extract_text_from_pdf(
    pdf_path: str,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
) -> list[dict]:
    """
    Extracts text page-by-page from a PDF file using pypdf.

    Preserves page numbering (1-indexed), metadata tags, empty-page safety,
    and returns a structured list of page dictionaries.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # If document_name is not provided, derive from filename
    if not document_name:
        document_name = path_obj.name

    reader = pypdf.PdfReader(str(path_obj))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        page_meta = {
            "page_number": page_number,
            "text": raw_text.strip(),
        }
        if document_id:
            page_meta["document_id"] = document_id
        if document_name:
            page_meta["document_name"] = document_name

        pages.append(page_meta)

    return pages
