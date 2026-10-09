import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.query_understanding_agent import QueryUnderstandingAgent


def test_validation_safety(agent: QueryUnderstandingAgent):
    print("--- Running Test 6: Malformed Output & Validation Safety ---")
    orig = "What is the policy?"

    # 1. Non-JSON string
    try:
        agent._parse_and_validate("This is not JSON", orig)
        assert False, "Failed to catch non-JSON string"
    except ValueError as e:
        assert "Failed to parse JSON" in str(e) or "Expecting" in str(e)

    # 2. Missing field
    bad_json_1 = json.dumps({"original_query": orig, "intent": "policy", "search_query": "policy"})
    try:
        agent._parse_and_validate(bad_json_1, orig)
        assert False, "Failed to catch missing keywords/filters"
    except ValueError as e:
        assert "Missing required field" in str(e)

    # 3. Bad type for keywords
    bad_json_2 = json.dumps({
        "original_query": orig,
        "intent": "policy",
        "search_query": "policy",
        "keywords": "just a string",
        "filters": {}
    })
    try:
        agent._parse_and_validate(bad_json_2, orig)
        assert False, "Failed to catch non-list keywords"
    except ValueError as e:
        assert "'keywords' must be a list" in str(e)

    # 4. Empty keywords list
    bad_json_3 = json.dumps({
        "original_query": orig,
        "intent": "policy",
        "search_query": "policy",
        "keywords": [],
        "filters": {}
    })
    try:
        agent._parse_and_validate(bad_json_3, orig)
        assert False, "Failed to catch empty keywords list"
    except ValueError as e:
        assert "'keywords' list cannot be empty" in str(e)

    # 5. Empty query check
    try:
        agent.analyze("")
        assert False, "Failed to catch empty input query"
    except ValueError as e:
        assert "Query cannot be empty" in str(e)

    print("Test 6 (Malformed Output Safety): Passed\n")


def test_agent_queries(agent: QueryUnderstandingAgent):
    # -------------------------------------------------------------
    # TEST 1: Annual Leave
    # -------------------------------------------------------------
    q1 = "How many days of annual leave do employees receive?"
    print("==================================================")
    print(f"--- TEST 1: {q1} ---")
    plan1 = agent.analyze(q1)
    print(json.dumps(plan1, indent=2))

    # Assertions
    assert plan1["original_query"] == q1, "original_query not preserved"
    assert "leave" in plan1["intent"].lower(), f"Intent does not mention leave: {plan1['intent']}"
    assert "leave" in plan1["search_query"].lower(), f"Search query does not mention leave: {plan1['search_query']}"
    assert len(plan1["keywords"]) >= 1, "Keywords list is empty"
    assert isinstance(plan1["filters"], dict), "Filters is not a dict"
    # Anti-answering check: Must not include factual answer "24 days"
    assert "24" not in plan1["intent"], "Agent answered the question with '24' in intent"
    assert "24" not in plan1["search_query"], "Agent answered the question with '24' in search_query"
    print("TEST 1 Passed.")

    # -------------------------------------------------------------
    # TEST 2: Remote Work
    # -------------------------------------------------------------
    q2 = "Can employees work from home three days a week?"
    print("\n==================================================")
    print(f"--- TEST 2: {q2} ---")
    plan2 = agent.analyze(q2)
    print(json.dumps(plan2, indent=2))

    assert plan2["original_query"] == q2, "original_query not preserved"
    intent_lower_2 = plan2["intent"].lower()
    search_lower_2 = plan2["search_query"].lower()
    assert "remote" in intent_lower_2 or "home" in intent_lower_2 or "work" in intent_lower_2
    assert "remote" in search_lower_2 or "home" in search_lower_2 or "work" in search_lower_2
    assert len(plan2["keywords"]) >= 1
    # Anti-answering check: Must not output affirmative answer or claim
    assert not plan2["intent"].lower().startswith("yes"), "Agent output affirmative answer"
    print("TEST 2 Passed.")

    # -------------------------------------------------------------
    # TEST 3: Password Policy
    # -------------------------------------------------------------
    q3 = "What is the password policy?"
    print("\n==================================================")
    print(f"--- TEST 3: {q3} ---")
    plan3 = agent.analyze(q3)
    print(json.dumps(plan3, indent=2))

    assert plan3["original_query"] == q3
    assert "password" in plan3["intent"].lower()
    assert "password" in plan3["search_query"].lower()
    assert any("password" in k.lower() for k in plan3["keywords"])
    print("TEST 3 Passed.")

    # -------------------------------------------------------------
    # TEST 4: Unsupported Topic (Parental Leave)
    # -------------------------------------------------------------
    q4 = "What is the parental leave duration?"
    print("\n==================================================")
    print(f"--- TEST 4: {q4} ---")
    plan4 = agent.analyze(q4)
    print(json.dumps(plan4, indent=2))

    assert plan4["original_query"] == q4
    assert "parental" in plan4["intent"].lower()
    assert "parental" in plan4["search_query"].lower()
    # Must not decide that evidence is insufficient (that belongs downstream)
    assert "insufficient" not in plan4["intent"].lower(), "Agent erroneously decided answer is unavailable"
    assert "not available" not in plan4["intent"].lower(), "Agent erroneously decided answer is unavailable"
    print("TEST 4 Passed.")

    # -------------------------------------------------------------
    # TEST 5: Schema and Type Integrity
    # -------------------------------------------------------------
    print("\n--- Running Test 5: Schema & Type Verification ---")
    for idx, plan in enumerate([plan1, plan2, plan3, plan4], start=1):
        assert isinstance(plan["original_query"], str)
        assert isinstance(plan["intent"], str)
        assert isinstance(plan["search_query"], str)
        assert isinstance(plan["keywords"], list)
        assert isinstance(plan["filters"], dict)
        assert len(plan["keywords"]) >= 2
    print("TEST 5 Passed.")


def main():
    print("Initializing QueryUnderstandingAgent...")
    agent = QueryUnderstandingAgent(model_name="llama3:latest")
    test_validation_safety(agent)
    test_agent_queries(agent)
    print("\nAll QueryUnderstandingAgent tests completed successfully!")


if __name__ == "__main__":
    main()
