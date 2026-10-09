from typing import Any, Optional


def calculate_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    """
    Computes a confusion matrix: matrix[true_label][pred_label] = count.
    """
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def calculate_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Computes accuracy, per-class Precision, Recall, F1, and Macro F1.
    Handles zero-division honestly by returning 0.0 and noting undefined status.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have identical length."
    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "per_class": {},
            "macro_f1": 0.0,
        }

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    per_class = {}
    f1_list = []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        is_precision_defined = (tp + fp) > 0
        is_recall_defined = (tp + fn) > 0

        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision_defined": is_precision_defined,
            "recall_defined": is_recall_defined,
        }

        if support > 0 or (tp + fp) > 0:
            f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": calculate_confusion_matrix(y_true, y_pred, labels),
    }


def calculate_source_page_recall(
    expected_sources: list[dict],
    predicted_citations: list[dict],
) -> dict[str, Any]:
    """
    Computes recall of expected (document_id, page_number) pairs against
    the predicted citations.
    """
    expected_set = {
        (s.get("document_id"), s.get("page_number"))
        for s in expected_sources
        if s.get("document_id") is not None and s.get("page_number") is not None
    }

    if not expected_set:
        return {
            "expected_count": 0,
            "matched_count": 0,
            "page_recall": 1.0,  # vacuously true if none expected
            "is_applicable": False,
        }

    predicted_set = {
        (c.get("document_id"), c.get("page_number"))
        for c in predicted_citations
        if c.get("document_id") is not None and c.get("page_number") is not None
    }

    matched = expected_set & predicted_set
    recall = len(matched) / len(expected_set)

    return {
        "expected_count": len(expected_set),
        "matched_count": len(matched),
        "page_recall": round(recall, 4),
        "is_applicable": True,
        "unmatched_sources": [
            {"document_id": doc, "page_number": pg}
            for doc, pg in (expected_set - matched)
        ],
    }


def calculate_citation_correctness(
    predicted_citations: list[dict],
    available_evidence: list[dict],
) -> dict[str, Any]:
    """
    Validates that each citation in predicted_citations corresponds to an actual
    chunk and page number present in the available evidence (hallucination check).
    """
    if not predicted_citations:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "hallucinated_citations": 0,
            "citation_validity_rate": 1.0,
        }

    valid_pairs = {
        (e.get("chunk_id"), e.get("page_number"))
        for e in available_evidence
        if e.get("chunk_id") is not None
    }

    valid_count = 0
    hallucinated = []

    for cit in predicted_citations:
        pair = (cit.get("chunk_id"), cit.get("page_number"))
        if pair in valid_pairs or any(
            cit.get("chunk_id") == e.get("chunk_id") for e in available_evidence
        ):
            valid_count += 1
        else:
            hallucinated.append(cit)

    validity_rate = valid_count / len(predicted_citations)

    return {
        "total_citations": len(predicted_citations),
        "valid_citations": valid_count,
        "hallucinated_citations": len(hallucinated),
        "citation_validity_rate": round(validity_rate, 4),
        "hallucinated_details": hallucinated,
    }
