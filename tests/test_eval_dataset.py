import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_status",
    "expected_sources",
    "expected_key_facts",
    "rationale",
}

VALID_STATUSES = {"grounded", "insufficient", "conflicting"}

REQUIRED_CATEGORIES = {
    "single_document_direct",
    "multi_document_synthesis",
    "insufficient_evidence",
    "compound_partial_support",
    "cross_document_conflict",
    "explicit_supersession",
    "citation_page_verification",
}


def validate_dataset(dataset_path: Path = DATASET_PATH) -> list[dict]:
    print(f"Validating dataset from: {dataset_path}")
    assert dataset_path.exists(), f"Evaluation dataset not found at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Evaluation dataset root must be a JSON array (list)."
    assert len(data) >= 20, f"Dataset must contain at least 20 evaluation examples, found {len(data)}."

    seen_ids = set()
    category_counts = {cat: 0 for cat in REQUIRED_CATEGORIES}

    for idx, item in enumerate(data):
        assert isinstance(item, dict), f"Item at index {idx} must be a dictionary."

        # Check required fields
        missing_fields = REQUIRED_FIELDS - set(item.keys())
        assert not missing_fields, f"Item {item.get('id', idx)} is missing required fields: {missing_fields}"

        # Check ID
        item_id = item["id"]
        assert isinstance(item_id, str) and item_id.strip(), f"Item {idx} has invalid id."
        assert item_id not in seen_ids, f"Duplicate ID detected: '{item_id}'"
        seen_ids.add(item_id)

        # Check category
        category = item["category"]
        assert category in REQUIRED_CATEGORIES, (
            f"Item {item_id} has invalid category '{category}'. Allowed: {REQUIRED_CATEGORIES}"
        )
        category_counts[category] += 1

        # Check question
        question = item["question"]
        assert isinstance(question, str) and len(question.strip()) >= 5, (
            f"Item {item_id} question must be a non-empty string."
        )

        # Check status
        status = item["expected_status"]
        assert status in VALID_STATUSES, (
            f"Item {item_id} has invalid expected_status '{status}'. Allowed: {VALID_STATUSES}"
        )

        # Check sources
        sources = item["expected_sources"]
        assert isinstance(sources, list), f"Item {item_id} expected_sources must be a list."
        for s_idx, s in enumerate(sources):
            assert isinstance(s, dict), f"Source {s_idx} in item {item_id} must be a dict."
            assert "document_id" in s, f"Source in {item_id} missing document_id."
            assert "page_number" in s and isinstance(s["page_number"], int), (
                f"Source in {item_id} must have integer page_number."
            )

        # Check key facts
        key_facts = item["expected_key_facts"]
        assert isinstance(key_facts, list), f"Item {item_id} expected_key_facts must be a list."

        # Status consistency checks
        if status == "grounded":
            assert len(sources) > 0, f"Grounded item {item_id} must specify expected_sources."
            assert len(key_facts) > 0, f"Grounded item {item_id} must specify expected_key_facts."
        elif status == "insufficient":
            assert len(sources) == 0, f"Insufficient item {item_id} should have empty expected_sources."
            assert len(key_facts) == 0, f"Insufficient item {item_id} should have empty expected_key_facts."
        elif status == "conflicting":
            assert len(sources) >= 2, f"Conflicting item {item_id} must specify at least 2 conflicting sources."

        # Check rationale
        rationale = item["rationale"]
        assert isinstance(rationale, str) and len(rationale.strip()) > 0, (
            f"Item {item_id} must provide a non-empty rationale."
        )

    # Check category distribution
    for cat, count in category_counts.items():
        assert count > 0, f"Category '{cat}' has zero test examples."

    print("\n--- Validation Summary ---")
    print(f"Total test examples: {len(data)}")
    print("Category breakdown:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")

    status_counts = {}
    for item in data:
        status_counts[item["expected_status"]] = status_counts.get(item["expected_status"], 0) + 1
    print("Status breakdown:")
    for status, count in status_counts.items():
        print(f"  - {status}: {count}")

    print("\nValidation passed successfully!")
    return data


def test_evaluation_dataset_schema():
    data = validate_dataset()
    assert len(data) >= 20


if __name__ == "__main__":
    validate_dataset()
