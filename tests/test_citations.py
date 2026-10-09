import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.citations import CitationFormatter
from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.rag_qa import RAGQuestionAnswerer


def run_unit_tests():
    formatter = CitationFormatter()
    doc_name = "PolicyPilot_Test_Enterprise_Handbook.pdf"

    # TEST 1 — Single citation
    test_1_input = {
        "chunk_id": "page-2-chunk-1",
        "page_number": 2,
        "text": "Employees receive 24 days of annual leave.",
        "similarity_score": 0.69,
    }
    c1 = formatter.format_citation(test_1_input, document_name=doc_name)
    assert c1 is not None, "Test 1 failed: Result was None"
    assert c1["page"] == 2, f"Test 1 failed: Expected page 2, got {c1['page']}"
    assert (
        c1["citation"] == "PolicyPilot Test Enterprise Handbook — Page 2"
    ), f"Test 1 failed: Got {c1['citation']}"
    print("TEST 1 (Single citation): Passed")

    # TEST 2 — Multiple citations
    test_2_inputs = [
        {"chunk_id": "page-2-chunk-1", "page_number": 2, "text": "Annual leave"},
        {"chunk_id": "page-3-chunk-1", "page_number": 3, "text": "Remote work"},
    ]
    c2 = formatter.format_citations(test_2_inputs, document_name=doc_name)
    assert len(c2) == 2, f"Test 2 failed: Expected 2 citations, got {len(c2)}"
    assert c2[0]["page"] == 2 and c2[1]["page"] == 3, "Test 2 failed: Page numbers mismatch"
    print("TEST 2 (Multiple citations): Passed")

    # TEST 3 — Duplicate page removal
    test_3_inputs = [
        {"chunk_id": "page-2-chunk-1", "page_number": 2, "text": "Chunk 1 on page 2"},
        {"chunk_id": "page-2-chunk-2", "page_number": 2, "text": "Chunk 2 on page 2"},
    ]
    c3 = formatter.format_citations(test_3_inputs, document_name=doc_name)
    assert len(c3) == 1, f"Test 3 failed: Expected 1 citation after deduplication, got {len(c3)}"
    assert c3[0]["page"] == 2, "Test 3 failed: Page should be 2"
    print("TEST 3 (Duplicate page removal): Passed")

    # TEST 4 — Missing page number
    test_4_input = {"chunk_id": "bad-chunk", "text": "No page"}
    c4 = formatter.format_citation(test_4_input, document_name=doc_name)
    assert c4 is None, f"Test 4 failed: Expected None for missing page_number, got {c4}"
    c4_list = formatter.format_citations([test_4_input], document_name=doc_name)
    assert len(c4_list) == 0, f"Test 4 list failed: Expected empty list, got {c4_list}"
    print("TEST 4 (Missing page number safety): Passed")

    # TEST 5 — No evidence / Insufficient evidence
    c5_empty = formatter.format_citations([], document_name=doc_name)
    assert len(c5_empty) == 0, "Test 5 failed: Expected 0 citations for empty results"
    c5_insufficient = formatter.format_citations(
        test_2_inputs, document_name=doc_name, answer="Insufficient evidence."
    )
    assert (
        len(c5_insufficient) == 0
    ), f"Test 5 failed: Expected 0 citations when answer is Insufficient evidence, got {c5_insufficient}"
    print("TEST 5 (No evidence & Insufficient evidence handling): Passed")


def run_real_document_test():
    print("\n--- Running Real PolicyPilot Document Citation Test ---")
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    )

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    embedding_model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    retriever = SemanticRetriever(vector_store)
    qa_engine = RAGQuestionAnswerer(model_name="llama3:latest")

    question = "How many days of annual leave do employees receive?"
    query_embedding = embedding_model.embed_text(question)

    retrieved = retriever.retrieve(query_embedding, chunks, top_k=3)

    response = qa_engine.answer_with_citations(
        question=question,
        retrieved_chunks=retrieved,
        document_name=pdf_path.name,
    )

    print(f"\nQuestion:\n{response['question']}")
    print(f"\nAnswer:\n{response['answer']}")
    print("\nSources:")
    for cit in response["citations"]:
        print(f"- {cit['citation']}")

    # Assertions
    assert "24" in response["answer"], "Factual answer must contain '24'"
    assert len(response["citations"]) > 0, "Expected at least one citation"
    first_citation = response["citations"][0]
    assert first_citation["page"] == 2, f"Expected top citation to be Page 2, got {first_citation['page']}"
    assert (
        first_citation["citation"] == "PolicyPilot Test Enterprise Handbook — Page 2"
    ), f"Unexpected citation format: {first_citation['citation']}"

    print("\nTEST 6 (Real PolicyPilot document): Passed")


def main():
    print("Running Citation Unit Tests...")
    run_unit_tests()
    run_real_document_test()
    print("\nAll Citation Tests Passed Successfully!")


if __name__ == "__main__":
    main()
