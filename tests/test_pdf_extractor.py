import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf


def test_handbook_pdf_extraction():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    assert pdf_path.exists(), f"PDF not found at {pdf_path}"

    pages = extract_text_from_pdf(str(pdf_path), document_id="DOC-HB-01")
    assert len(pages) == 7

    # Verify 1-indexed numbering and metadata
    for idx, page in enumerate(pages, start=1):
        assert page["page_number"] == idx
        assert page["document_id"] == "DOC-HB-01"
        assert page["document_name"] == "PolicyPilot_Test_Enterprise_Handbook.pdf"
        assert len(page["text"]) > 0

    # Verify specific policy sections on key pages
    assert "Annual Leave Policy" in pages[1]["text"]
    assert "Remote Work Policy" in pages[2]["text"]


def test_remote_work_addendum_extraction():
    pdf_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"
    assert pdf_path.exists(), f"PDF not found at {pdf_path}"

    pages = extract_text_from_pdf(str(pdf_path), document_id="DOC-RW-02")
    assert len(pages) == 2

    for idx, page in enumerate(pages, start=1):
        assert page["page_number"] == idx
        assert page["document_id"] == "DOC-RW-02"
        assert page["document_name"] == "PolicyPilot_Test_Remote_Work_Addendum.pdf"
        assert len(page["text"]) > 0


def test_missing_pdf_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("nonexistent_path_to_policy.pdf")


def main():
    test_handbook_pdf_extraction()
    test_remote_work_addendum_extraction()
    test_missing_pdf_raises_file_not_found()
    print("All PDF extraction tests passed successfully.")


if __name__ == "__main__":
    main()
