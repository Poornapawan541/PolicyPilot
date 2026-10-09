from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from app.embedding.embedder import EmbeddingModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.agents.query_understanding_agent import QueryUnderstandingAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.evidence_verification_agent import EvidenceVerificationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.response_agent import ResponseAgent


from app.core.tracer import WorkflowTracer, derive_status_reason


class PolicyPilotState(TypedDict):
    """
    Workflow state for the PolicyPilot LangGraph orchestration layer.
    """
    original_query: str
    query_plan: Optional[dict]
    evidence: list[dict]
    verification_result: Optional[dict]
    reasoning_result: Optional[dict]
    response_result: Optional[dict]
    final_answer: Optional[str]
    status: Optional[str]
    error: Optional[str]
    trace_id: Optional[str]
    enable_tracing: Optional[bool]
    trace: Optional[dict]


def route_after_verification(state: PolicyPilotState) -> str:
    """
    Deterministic routing function after evidence verification.

    Routes:
    - 'sufficient' -> 'reasoning'
    - 'insufficient' -> 'response'
    - 'conflicting' -> 'response'
    - any error/other -> 'response' (fail closed safely)
    """
    if state.get("error"):
        return "response"

    ver_res = state.get("verification_result")
    if not ver_res or not isinstance(ver_res, dict):
        return "response"

    ver_status = ver_res.get("status")
    if ver_status == "sufficient":
        return "reasoning"
    elif ver_status in ("insufficient", "conflicting"):
        return "response"
    else:
        return "response"


def build_policypilot_graph(
    chunks: list[dict],
    query_agent: Optional[QueryUnderstandingAgent] = None,
    retrieval_agent: Optional[RetrievalAgent] = None,
    verification_agent: Optional[EvidenceVerificationAgent] = None,
    reasoning_agent: Optional[ReasoningAgent] = None,
    response_agent: Optional[ResponseAgent] = None,
):
    """
    Builds the end-to-end PolicyPilot LangGraph workflow with conditional routing.

    Flow:
    START -> query_understanding -> retrieval -> verification
                                                    |
                       +----------------------------+----------------------------+
                       |                                                         |
                   sufficient                                          insufficient / conflicting
                       |                                                         |
                       v                                                         |
                   reasoning                                                     |
                       |                                                         |
                       +----------------------------+----------------------------+
                                                    |
                                                    v
                                                 response -> END
    """
    # Instantiate agents with dependency injection if not provided
    if query_agent is None:
        query_agent = QueryUnderstandingAgent()

    if retrieval_agent is None:
        from app.retrieval.vector_store import VectorStore
        from app.retrieval.retriever import SemanticRetriever
        from app.retrieval.bm25_retriever import BM25Retriever

        embedder = EmbeddingModel()
        texts = [c["text"] for c in chunks] if chunks else []
        embeddings = embedder.embed_texts(texts) if texts else []
        vector_store = VectorStore(dimension=384)
        if embeddings:
            vector_store.add_embeddings(embeddings)

        semantic_retriever = SemanticRetriever(vector_store)
        bm25_retriever = BM25Retriever(chunks if chunks else [])
        hybrid_retriever = HybridRetriever(semantic_retriever, bm25_retriever)
        reranker = CrossEncoderReranker()
        retrieval_agent = RetrievalAgent(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            embedding_model=embedder,
        )

    if verification_agent is None:
        verification_agent = EvidenceVerificationAgent()

    if reasoning_agent is None:
        reasoning_agent = ReasoningAgent()

    if response_agent is None:
        response_agent = ResponseAgent()

    # Helper to retrieve or initialize tracer from state
    def get_tracer(state: PolicyPilotState) -> WorkflowTracer:
        existing_trace = state.get("trace")
        trace_id = state.get("trace_id")
        enabled = state.get("enable_tracing")
        if existing_trace and isinstance(existing_trace, dict) and "_tracer_obj" in existing_trace:
            return existing_trace["_tracer_obj"]

        tracer = WorkflowTracer(trace_id=trace_id, enabled=enabled)
        return tracer

    # Define Node 1: query_understanding
    def query_understanding_node(state: PolicyPilotState) -> dict:
        tracer = get_tracer(state)
        tracer.start_stage("query_understanding")
        try:
            query = state.get("original_query", "")
            if not query or not query.strip():
                err_msg = "original_query must be a non-empty string."
                tracer.end_stage("query_understanding", error=err_msg)
                tracer.record_decision("error", reason="Empty query provided.", error=err_msg)
                return {
                    "error": err_msg,
                    "status": "error",
                    "final_answer": "Query cannot be empty.",
                    "trace_id": tracer.trace_id,
                    "trace": {**tracer.finish(), "_tracer_obj": tracer},
                }
            plan = query_agent.analyze(query)
            meta = {
                "intent": plan.get("intent") if isinstance(plan, dict) else None,
                "keywords": plan.get("keywords") if isinstance(plan, dict) else None,
                "has_filters": bool(plan.get("filters")) if isinstance(plan, dict) else False,
            }
            tracer.end_stage("query_understanding", metadata=meta)
            return {
                "query_plan": plan,
                "error": None,
                "trace_id": tracer.trace_id,
                "trace": {"_tracer_obj": tracer},
            }
        except Exception as e:
            err_msg = f"query_understanding node failed: {str(e)}"
            tracer.end_stage("query_understanding", error=err_msg)
            tracer.record_decision("error", reason=err_msg, error=err_msg)
            return {
                "error": err_msg,
                "status": "error",
                "final_answer": "An error occurred during query understanding.",
                "trace_id": tracer.trace_id,
                "trace": {**tracer.finish(), "_tracer_obj": tracer},
            }

    # Define Node 2: retrieval
    def retrieval_node(state: PolicyPilotState) -> dict:
        tracer = get_tracer(state)
        if state.get("error"):
            return {}
        tracer.start_stage("retrieval")
        try:
            plan = state.get("query_plan")
            if not plan:
                err_msg = "No query plan found for retrieval node."
                tracer.end_stage("retrieval", error=err_msg)
                return {
                    "evidence": [],
                    "error": err_msg,
                    "status": "error",
                    "trace": {"_tracer_obj": tracer},
                }
            retrieved_evidence = retrieval_agent.retrieve(plan, chunks=chunks, top_k=3)
            meta = {
                "evidence": retrieved_evidence,
                "retrieved_count": len(retrieved_evidence),
            }
            tracer.end_stage("retrieval", metadata=meta)
            return {
                "evidence": retrieved_evidence,
                "error": None,
                "trace": {"_tracer_obj": tracer},
            }
        except Exception as e:
            err_msg = f"retrieval node failed: {str(e)}"
            tracer.end_stage("retrieval", error=err_msg)
            return {
                "evidence": [],
                "error": err_msg,
                "status": "error",
                "final_answer": "An error occurred during retrieval.",
                "trace": {"_tracer_obj": tracer},
            }

    # Define Node 3: verification
    def verification_node(state: PolicyPilotState) -> dict:
        tracer = get_tracer(state)
        if state.get("error"):
            return {}
        tracer.start_stage("verification")
        try:
            query = state.get("original_query", "")
            evidence = state.get("evidence", [])
            ver_res = verification_agent.verify(query, evidence)
            meta = {
                "verification_status": ver_res.get("status") if isinstance(ver_res, dict) else None,
                "conflicts_count": len(ver_res.get("conflicts", [])) if isinstance(ver_res, dict) else 0,
                "missing_aspects": ver_res.get("missing_aspects", []) if isinstance(ver_res, dict) else [],
            }
            tracer.end_stage("verification", metadata=meta)
            return {
                "verification_result": ver_res,
                "error": None,
                "trace": {"_tracer_obj": tracer},
            }
        except Exception as e:
            err_msg = f"verification node failed: {str(e)}"
            tracer.end_stage("verification", error=err_msg)
            return {
                "error": err_msg,
                "status": "error",
                "final_answer": "An error occurred during evidence verification.",
                "trace": {"_tracer_obj": tracer},
            }

    # Define Node 4: reasoning
    def reasoning_node(state: PolicyPilotState) -> dict:
        tracer = get_tracer(state)
        if state.get("error"):
            return {}
        tracer.start_stage("reasoning")
        try:
            query = state.get("original_query", "")
            ver_res = state.get("verification_result", {})
            evidence = state.get("evidence", [])
            reas_res = reasoning_agent.reason(query, ver_res, evidence)
            meta = {
                "reasoning_status": reas_res.get("status") if isinstance(reas_res, dict) else None,
                "evidence_used_count": len(reas_res.get("evidence_used", [])) if isinstance(reas_res, dict) else 0,
            }
            tracer.end_stage("reasoning", metadata=meta)
            return {
                "reasoning_result": reas_res,
                "status": reas_res.get("status"),
                "error": None,
                "trace": {"_tracer_obj": tracer},
            }
        except Exception as e:
            err_msg = f"reasoning node failed: {str(e)}"
            tracer.end_stage("reasoning", error=err_msg)
            return {
                "error": err_msg,
                "status": "error",
                "final_answer": "An error occurred during reasoning.",
                "trace": {"_tracer_obj": tracer},
            }

    # Define Node 5: response
    def response_node(state: PolicyPilotState) -> dict:
        tracer = get_tracer(state)
        tracer.start_stage("response")
        if state.get("error"):
            tracer.end_stage("response", error=state.get("error"))
            tracer.record_decision("error", reason=state.get("error"), error=state.get("error"))
            final_trace = tracer.finish()
            return {
                "status": "error",
                "trace_id": tracer.trace_id,
                "trace": final_trace,
            }
        try:
            query = state.get("original_query", "")
            reas_res = state.get("reasoning_result")
            evidence = state.get("evidence", [])
            ver_res = state.get("verification_result", {})

            # If reasoning_result was bypassed because of insufficient or conflicting verification:
            if not reas_res:
                ver_status = ver_res.get("status", "insufficient") if ver_res else "insufficient"
                reas_res = {
                    "status": ver_status,
                    "conclusion": None,
                    "evidence_used": [],
                    "conflicts": ver_res.get("conflicts", []) if ver_res else [],
                }

            resp_res = response_agent.respond(query, reas_res, evidence)
            final_status = resp_res.get("status")
            meta = {
                "response_status": final_status,
                "citations_count": len(resp_res.get("citations", [])) if isinstance(resp_res, dict) else 0,
            }
            tracer.end_stage("response", metadata=meta)

            # Record final decision explanation
            reason = derive_status_reason(final_status, ver_res, reas_res)
            tracer.record_decision(final_status=final_status, reason=reason)
            final_trace = tracer.finish()

            return {
                "reasoning_result": state.get("reasoning_result"),
                "response_result": resp_res,
                "final_answer": resp_res.get("answer"),
                "status": final_status,
                "error": None,
                "trace_id": tracer.trace_id,
                "trace": final_trace,
            }
        except Exception as e:
            err_msg = f"response node failed: {str(e)}"
            tracer.end_stage("response", error=err_msg)
            tracer.record_decision("error", reason=err_msg, error=err_msg)
            final_trace = tracer.finish()
            return {
                "error": err_msg,
                "status": "error",
                "final_answer": "An error occurred during response generation.",
                "trace_id": tracer.trace_id,
                "trace": final_trace,
            }

    # Construct LangGraph StateGraph
    builder = StateGraph(PolicyPilotState)

    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("verification", verification_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("response", response_node)

    # Graph Flow:
    # START -> query_understanding -> retrieval -> verification
    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "retrieval")
    builder.add_edge("retrieval", "verification")

    # Conditional routing after verification:
    # "sufficient" -> "reasoning"
    # "insufficient" -> "response"
    # "conflicting" -> "response"
    builder.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "reasoning": "reasoning",
            "response": "response",
        },
    )

    # reasoning -> response
    builder.add_edge("reasoning", "response")

    # response -> END
    builder.add_edge("response", END)

    return builder.compile()
