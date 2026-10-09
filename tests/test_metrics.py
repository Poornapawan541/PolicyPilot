import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_source_page_recall,
    calculate_citation_correctness,
    calculate_confusion_matrix,
)


def test_perfect_classification():
    y_true = ["grounded", "insufficient", "conflicting"]
    y_pred = ["grounded", "insufficient", "conflicting"]
    res = calculate_classification_metrics(y_true, y_pred)

    assert res["total_samples"] == 3
    assert res["correct_predictions"] == 3
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    for label in ["grounded", "insufficient", "conflicting"]:
        assert res["per_class"][label]["precision"] == 1.0
        assert res["per_class"][label]["recall"] == 1.0
        assert res["per_class"][label]["f1"] == 1.0


def test_zero_division_honest_handling():
    # Model never predicts "conflicting", and predicts "insufficient" when none are true
    y_true = ["grounded", "grounded"]
    y_pred = ["insufficient", "insufficient"]
    labels = ["grounded", "insufficient", "conflicting"]
    res = calculate_classification_metrics(y_true, y_pred, labels=labels)

    assert res["accuracy"] == 0.0
    assert res["macro_f1"] == 0.0

    # Grounded: support 2, predicted 0 -> precision 0.0 (undefined), recall 0.0
    assert res["per_class"]["grounded"]["precision"] == 0.0
    assert res["per_class"]["grounded"]["precision_defined"] is False
    assert res["per_class"]["grounded"]["recall"] == 0.0

    # Insufficient: support 0, predicted 2 -> precision 0.0, recall 0.0 (undefined)
    assert res["per_class"]["insufficient"]["precision"] == 0.0
    assert res["per_class"]["insufficient"]["recall"] == 0.0
    assert res["per_class"]["insufficient"]["recall_defined"] is False

    # Conflicting: support 0, predicted 0 -> both undefined
    assert res["per_class"]["conflicting"]["precision"] == 0.0
    assert res["per_class"]["conflicting"]["precision_defined"] is False
    assert res["per_class"]["conflicting"]["recall"] == 0.0
    assert res["per_class"]["conflicting"]["recall_defined"] is False


def test_confusion_matrix():
    y_true = ["grounded", "grounded", "insufficient"]
    y_pred = ["grounded", "insufficient", "insufficient"]
    labels = ["grounded", "insufficient"]
    cm = calculate_confusion_matrix(y_true, y_pred, labels)

    assert cm["grounded"]["grounded"] == 1
    assert cm["grounded"]["insufficient"] == 1
    assert cm["insufficient"]["insufficient"] == 1
    assert cm["insufficient"]["grounded"] == 0


def test_source_page_recall():
    expected = [
        {"document_id": "DOC-A", "page_number": 2},
        {"document_id": "DOC-B", "page_number": 5},
    ]

    # Perfect recall
    predicted_perfect = [
        {"document_id": "DOC-A", "page_number": 2},
        {"document_id": "DOC-B", "page_number": 5},
    ]
    res1 = calculate_source_page_recall(expected, predicted_perfect)
    assert res1["page_recall"] == 1.0
    assert res1["matched_count"] == 2
    assert res1["unmatched_sources"] == []

    # Partial recall (1 out of 2)
    predicted_partial = [
        {"document_id": "DOC-A", "page_number": 2},
        {"document_id": "DOC-C", "page_number": 1},
    ]
    res2 = calculate_source_page_recall(expected, predicted_partial)
    assert res2["page_recall"] == 0.5
    assert res2["matched_count"] == 1
    assert len(res2["unmatched_sources"]) == 1

    # Empty expected (e.g. insufficient)
    res_empty = calculate_source_page_recall([], predicted_perfect)
    assert res_empty["is_applicable"] is False


def test_citation_correctness_hallucination_check():
    evidence = [
        {"chunk_id": "c1", "page_number": 2, "text": "hello"},
        {"chunk_id": "c2", "page_number": 3, "text": "world"},
    ]

    # Valid citations
    valid_cits = [
        {"chunk_id": "c1", "page_number": 2},
    ]
    res_val = calculate_citation_correctness(valid_cits, evidence)
    assert res_val["citation_validity_rate"] == 1.0
    assert res_val["hallucinated_citations"] == 0

    # Hallucinated citation
    invalid_cits = [
        {"chunk_id": "c1", "page_number": 2},
        {"chunk_id": "c99", "page_number": 99},
    ]
    res_inval = calculate_citation_correctness(invalid_cits, evidence)
    assert res_inval["total_citations"] == 2
    assert res_inval["valid_citations"] == 1
    assert res_inval["hallucinated_citations"] == 1
    assert res_inval["citation_validity_rate"] == 0.5
    assert len(res_inval["hallucinated_details"]) == 1


if __name__ == "__main__":
    test_perfect_classification()
    test_zero_division_honest_handling()
    test_confusion_matrix()
    test_source_page_recall()
    test_citation_correctness_hallucination_check()
    print("All metric calculation unit tests passed!")
