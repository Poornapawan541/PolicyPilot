import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf


def main():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    pages = extract_text_from_pdf(str(pdf_path))

    print(f"Total pages extracted: {len(pages)}")

    if pages:
        first_page = pages[0]
        print(f"\n--- First Page (Page {first_page['page_number']}) ---")
        print(first_page["text"][:300])

        last_page = pages[-1]
        print(f"\n--- Last Page (Page {last_page['page_number']}) ---")
        print(last_page["text"][:200])


if __name__ == "__main__":
    main()
