"""Advisor Agent: Agentic RAG for curriculum Q&A (Sprint 2) + grounded placement lookup.

Curriculum mapping: "Agentic RAG & self-reflection". Flow:
retrieve -> reflect (is context enough?) -> [reformulate -> retrieve]* -> answer.
If context is still insufficient after MAX_RAG_REFORMULATIONS the agent says so
instead of answering from model memory (hallucination guard).
"""
from __future__ import annotations

from src.agents.messages import ai_msg, tool_msg
from src.config import MAX_RAG_REFORMULATIONS, UNABLE_TO_VERIFY
from src.llm import get_llm, invoke_structured
from src.schemas import ANSWER_PROMPT, REFLECT_PROMPT, ContextGrade
from src.state import OrchestratorState
from src.tools.db_queries import match_partners
from src.tools.rag_retriever import retrieve_curriculum


def advisor_retrieve(state: OrchestratorState) -> dict:
    """Retrieve curriculum passages for the (possibly reformulated) query."""
    query = state.get("rag_query") or state["question"]
    result = retrieve_curriculum(query)
    return {"rag_passages": result.get("passages", []), "rag_query": query,
            "context_sufficient": result["status"] == "ok" and bool(result.get("passages")),
            "messages": [tool_msg("rag_retriever", {k: v for k, v in result.items() if k != "passages"}
                                  | {"n_passages": len(result.get("passages", []))})]}


def advisor_reflect(state: OrchestratorState) -> dict:
    """Self-reflection: LLM judges whether retrieved context can answer the question."""
    passages = state.get("rag_passages", [])
    if not passages:
        return {"context_sufficient": False, "rag_attempts": state.get("rag_attempts", 0) + 1}
    ctx = "\n---\n".join(p["text"] for p in passages)
    grade = invoke_structured(ContextGrade, REFLECT_PROMPT.format(question=state["question"], context=ctx))
    return {"context_sufficient": grade.sufficient,
            "rag_query": grade.reformulated_query or state.get("rag_query", ""),
            "rag_attempts": state.get("rag_attempts", 0) + 1}


def route_after_reflect(state: OrchestratorState) -> str:
    """Conditional edge: answer if sufficient/out of retries, else search again."""
    if state.get("context_sufficient") or state.get("rag_attempts", 0) > MAX_RAG_REFORMULATIONS:
        return "advisor_answer"
    return "advisor_retrieve"


def advisor_answer(state: OrchestratorState) -> dict:
    """Answer strictly from retrieved excerpts, or admit the curriculum lacks the answer."""
    if not state.get("context_sufficient"):
        text = ("I couldn't find this in the Sprints curriculum, so I won't guess. "
                "Please ask a human advisor.")
    else:
        ctx = "\n---\n".join(f"[{p['source']}] {p['text']}" for p in state["rag_passages"])
        text = get_llm().invoke(ANSWER_PROMPT.format(context=ctx, question=state["question"])).content
    return {"final_answer": text, "messages": [ai_msg("advisor", text)]}


def placement_lookup(state: OrchestratorState) -> dict:
    """Fetch live DB records and compute partner matches (deterministic, no LLM)."""
    result = match_partners(state["learner_id"])
    return {"placement": result, "messages": [tool_msg("match_partners", result)]}


def route_after_placement(state: OrchestratorState) -> str:
    """Tier-1 recommendations must pause for a human (HITL); others finalize directly."""
    p = state["placement"]
    if p["status"] == "ok" and any(r["tier1"] for r in p["recommendations"]):
        return "hitl_gate"
    return "placement_finalize"
