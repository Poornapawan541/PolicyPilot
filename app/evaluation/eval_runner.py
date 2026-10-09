"""
Step 21: Reproducible Evaluation Runner for PolicyPilot.

Executes the complete evaluation dataset against the PolicyPilot workflow graph,
records actual predictions alongside expected labels, computes precision/recall/F1/accuracy
and source page recall, and writes machine-readable results under data/evaluation/results/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_extractor import extract_text_from_pdf
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import SemanticRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent
from app.workflow.policypilot_graph import build_policypilot_graph
from app.evaluation.fixtures import get_evaluation_fixture_chunks
from app.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_source_page_recall,
    calculate_citation_correctness,
)


def load_base_corpus() -> list[dict]:
    doc1_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Enterprise_Handbook.pdf"
    doc2_path = PROJECT_ROOT / "data" / "documents" / "PolicyPilot_Test_Remote_Work_Addendum.pdf"

    pages1 = extract_text_from_pdf(
        str(doc1_path),
        document_id="PP-TEST-001",
        document_name="PolicyPilot_Test_Enterprise_Handbook.pdf",
    )
    chunks1 = chunk_pages(pages1, chunk_size=200, overlap=30)

    pages2 = extract_text_from_pdf(
        str(doc2_path),
        document_id="PP-TEST-002",
        document_name="PolicyPilot_Test_Remote_Work_Addendum.pdf",
    )
    chunks2 = chunk_pages(pages2, chunk_size=200, overlap=30)

    return chunks1 + chunks2


def create_pipeline(chunks: list[dict], embedder: EmbeddingModel, reranker: CrossEncoderReranker):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    vector_store = VectorStore(dimension=384)
    vector_store.add_embeddings(embeddings)

    semantic_retriever = SemanticRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    retrieval_agent = RetrievalAgent(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        embedding_model=embedder,
    )
    verification_agent = EvidenceVerificationAgent()
    reasoning_agent = ReasoningAgent()
    response_agent = ResponseAgent()

    graph = build_policypilot_graph(
        chunks=chunks,
        retrieval_agent=retrieval_agent,
        verification_agent=verification_agent,
        reasoning_agent=reasoning_agent,
        response_agent=response_agent,
    )
    return graph


def run_evaluation(
    dataset_path: Path = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json",
    output_dir: Path = PROJECT_ROOT / "data" / "evaluation" / "results",
    sample_limit: int | None = None,
) -> dict[str, Any]:
    print("=" * 60)
    print("STARTING POLICYPILOT EVALUATION RUN (STEP 21)")
    print("=" * 60)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if sample_limit is not None:
        dataset = dataset[:sample_limit]
        print(f"Limiting evaluation to first {sample_limit} items.")

    base_chunks = load_base_corpus()
    print(f"Loaded base corpus: {len(base_chunks)} chunks from 2 documents.")

    embedder = EmbeddingModel()
    reranker = CrossEncoderReranker()

    # Pre-build base graph
    base_graph = create_pipeline(base_chunks, embedder, reranker)

    results = []
    y_true = []
    y_pred = []
    page_recalls = []
    citation_validities = []

    start_time = time.time()

    for idx, item in enumerate(dataset):
        eval_id = item["id"]
        category = item["category"]
        question = item["question"]
        expected_status = item["expected_status"]
        expected_sources = item.get("expected_sources", [])

        print(f"\n[{idx+1}/{len(dataset)}] Evaluating {eval_id} ({category}):")
        print(f"  Q: {question}")
        print(f"  Expected: {expected_status}")

        # Check if item requires conflict/supersession fixture chunks
        fixture_chunks = get_evaluation_fixture_chunks(eval_id)
        if fixture_chunks:
            corpus_with_fixtures = base_chunks + fixture_chunks
            graph = create_pipeline(corpus_with_fixtures, embedder, reranker)
        else:
            graph = base_graph

        item_start = time.time()
        try:
            prediction = graph.invoke({"original_query": question, "evidence": []})
            pred_status = prediction.get("status", "error")
            pred_answer = prediction.get("final_answer", "")
            pred_citations = prediction.get("response_result", {}).get("citations", [])
            retrieved_evidence = prediction.get("evidence", [])
            item_error = None
        except Exception as e:
            pred_status = "error"
            pred_answer = ""
            pred_citations = []
            retrieved_evidence = []
            item_error = str(e)
        item_duration = round(time.time() - item_start, 2)

        is_status_correct = (pred_status == expected_status)
        y_true.append(expected_status)
        y_pred.append(pred_status)

        # Page recall (when expected grounded)
        page_recall_info = calculate_source_page_recall(expected_sources, pred_citations)
        if page_recall_info["is_applicable"]:
            page_recalls.append(page_recall_info["page_recall"])

        # Citation validity
        cit_info = calculate_citation_correctness(pred_citations, retrieved_evidence)
        citation_validities.append(cit_info["citation_validity_rate"])

        print(f"  Predicted: {pred_status} (Correct: {is_status_correct}) in {item_duration}s")
        if pred_citations:
            print(f"  Citations: {[c.get('citation') for c in pred_citations]}")

        results.append({
            "id": eval_id,
            "category": category,
            "question": question,
            "expected_status": expected_status,
            "predicted_status": pred_status,
            "is_status_correct": is_status_correct,
            "expected_sources": expected_sources,
            "predicted_citations": pred_citations,
            "predicted_answer": pred_answer,
            "page_recall": page_recall_info,
            "citation_validity": cit_info,
            "duration_seconds": item_duration,
            "error": item_error,
        })

    total_duration = round(time.time() - start_time, 2)

    # Calculate overall metrics
    all_statuses = ["grounded", "insufficient", "conflicting"]
    classification_metrics = calculate_classification_metrics(y_true, y_pred, labels=all_statuses)

    mean_page_recall = round(sum(page_recalls) / len(page_recalls), 4) if page_recalls else 0.0
    mean_citation_validity = round(sum(citation_validities) / len(citation_validities), 4) if citation_validities else 1.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_evaluated": len(dataset),
        "total_duration_seconds": total_duration,
        "classification_metrics": classification_metrics,
        "source_page_recall": {
            "evaluated_grounded_samples": len(page_recalls),
            "mean_page_recall": mean_page_recall,
        },
        "citation_validity": {
            "evaluated_samples": len(citation_validities),
            "mean_validity_rate": mean_citation_validity,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION RUN COMPLETE")
    print("=" * 60)
    print(f"Total Evaluated: {len(dataset)}")
    print(f"Total Duration: {total_duration}s")
    print(f"Overall Status Accuracy: {classification_metrics['accuracy'] * 100:.1f}%")
    print(f"Macro F1 Score: {classification_metrics['macro_f1']:.4f}")
    print("Per-class performance:")
    for cls_name, cls_m in classification_metrics["per_class"].items():
        print(f"  - {cls_name}: Precision={cls_m['precision']}, Recall={cls_m['recall']}, F1={cls_m['f1']} (support={cls_m['support']})")
    print(f"Mean Source Page Recall (Grounded): {mean_page_recall * 100:.1f}%")
    print(f"Citation Correctness: {mean_citation_validity * 100:.1f}%")
    print(f"Results saved to: {results_path}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    run_evaluation()
