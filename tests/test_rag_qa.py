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
from app.retrieval.rag_qa import RAGQuestionAnswerer


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

    # 1. Ingestion: Extract & Chunk
    print("Ingesting document...")
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)
    print(f"Extracted {len(pages)} pages into {len(chunks)} chunks.")

    # 2. Embedding & Indexing
    print("Generating embeddings and building FAISS index...")
    embedding_model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    retriever = SemanticRetriever(vector_store)
    qa_engine = RAGQuestionAnswerer(model_name="llama3:latest")

    # ---------------------------------------------------------
    # TEST 1: Supported Question
    # ---------------------------------------------------------
    question_1 = "How many days of annual leave do employees receive?"
    print("\n==================================================")
    print("--- Supported Question ---")
    print(f"Question:\n{question_1}")

    q1_embedding = embedding_model.embed_text(question_1)
    retrieved_1 = retriever.retrieve(q1_embedding, chunks, top_k=3)

    print("\nRetrieved evidence:")
    for r in retrieved_1:
        print(f"- [Chunk: {r['chunk_id']} | Page: {r['page_number']} | Score: {r['similarity_score']:.4f}]: {r['text'][:120]}...")

    answer_1 = qa_engine.answer_question(question_1, retrieved_1)
    print(f"\nGenerated answer:\n{answer_1}")

    if "24" not in answer_1:
        print("\nError: Expected '24' to be present in answer for annual leave.")
        return

    # ---------------------------------------------------------
    # TEST 2: Unsupported Question (Insufficient Evidence)
    # ---------------------------------------------------------
    question_2 = "How many weeks of parental leave do employees receive?"
    print("\n==================================================")
    print("--- Unsupported Question ---")
    print(f"Question:\n{question_2}")

    q2_embedding = embedding_model.embed_text(question_2)
    retrieved_2 = retriever.retrieve(q2_embedding, chunks, top_k=3)

    print("\nRetrieved evidence:")
    for r in retrieved_2:
        print(f"- [Chunk: {r['chunk_id']} | Page: {r['page_number']} | Score: {r['similarity_score']:.4f}]: {r['text'][:120]}...")

    answer_2 = qa_engine.answer_question(question_2, retrieved_2)
    print(f"\nGenerated answer:\n{answer_2}")

    lower_ans_2 = answer_2.lower()
    if "insufficient evidence" not in lower_ans_2 and "does not specify" not in lower_ans_2 and "not specified" not in lower_ans_2:
        print("\nError: Expected answer to reflect insufficient evidence.")
        return

    print("\n==================================================")
    print("RAG QA test passed successfully.")


if __name__ == "__main__":
    main()
