"""Orchestrator: ONE LangGraph StateGraph routing to Advisor / Assessor workers (Sprint 2).

Features: TypedDict global state, intent routing, MemorySaver per-learner threads,
Agentic RAG loop, Evaluator-Optimizer loop, HITL interrupt for tier-1 partners.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents import advisor_agent as adv
from src.agents import assessor_agent as ass
from src.agents.messages import ai_msg
from src.llm import invoke_structured
from src.schemas import ROUTER_PROMPT, RouteDecision
from src.state import OrchestratorState

_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_PLACEMENT_WORDS = ("placement", "job", "opening", "match", "hire", "partner", "vacancy")


def heuristic_intent(text: str, has_code: bool) -> str:
    """Deterministic fallback router (used if the LLM router fails)."""
    if has_code:
        return "code_submission"
    if any(w in text.lower() for w in _PLACEMENT_WORDS):
        return "placement_request"
    return "curriculum_question"


def classify_intent(state: OrchestratorState) -> dict:
    """Orchestrator node: extract code/question from the last human message and route by intent."""
    last = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)).content
    fence = _CODE_FENCE.search(last)
    code = fence.group(1) if fence else state.get("code", "")
    text = _CODE_FENCE.sub("", last).strip()
    try:
        intent = invoke_structured(RouteDecision, ROUTER_PROMPT.format(message=last)).intent
        if code:                       # code present => always grade it
            intent = "code_submission"
    except Exception:
        intent = heuristic_intent(text, bool(code))
    return {"intent": intent, "question": text, "code": code, "rag_query": "", "rag_attempts": 0,
            "messages": [ai_msg("orchestrator", f"Routing to {intent} worker")]}


def route_by_intent(state: OrchestratorState) -> str:
    """Conditional edge from the orchestrator to the right worker."""
    return {"curriculum_question": "advisor_retrieve", "code_submission": "assessor_execute",
            "placement_request": "placement_lookup"}[state["intent"]]


def hitl_gate(state: OrchestratorState) -> dict:
    """Pause the graph for human approval of tier-1 (e.g. Fawry) placements.

    interrupt() persists state via the checkpointer; execution resumes only when the
    caller sends Command(resume={"approved": bool, "reviewer": str, "note": str}).
    """
    tier1 = [r for r in state["placement"]["recommendations"] if r["tier1"]]
    decision = interrupt({
        "type": "placement_approval", "learner": state["placement"]["learner"],
        "tier1_recommendations": tier1,
        "question": "Approve releasing these tier-1 partner recommendations to the learner?"})
    return {"hitl": decision}


def placement_finalize(state: OrchestratorState) -> dict:
    """Compose the placement message from DB facts via a template (zero LLM freedom)."""
    p, hitl = state["placement"], state.get("hitl")
    if p["status"] not in ("ok",):
        text = p.get("message") or "No eligible partner opening matched your current profile."
    else:
        recs = p["recommendations"]
        if hitl is not None and not hitl.get("approved"):
            recs = [r for r in recs if not r["tier1"]]   # human rejected tier-1 items
        if not recs:
            text = "A human reviewer did not approve the recommendation; your advisor will follow up."
        else:
            who = p["learner"]["name"]
            lines = [f"- {r['employer']}: {r['title']} ({r['salary_min_egp']:,}-{r['salary_max_egp']:,} EGP)"
                     + (" [approved by human reviewer]" if r["tier1"] and hitl else "") for r in recs]
            text = f"Verified matches for {who} from live Sprints records:\n" + "\n".join(lines)
    return {"final_answer": text, "messages": [ai_msg("advisor", text)]}


def build_graph(checkpointer=None):
    """Assemble and compile the orchestration graph (MemorySaver by default)."""
    g = StateGraph(OrchestratorState)
    g.add_node("classify_intent", classify_intent)
    for name, fn in [("advisor_retrieve", adv.advisor_retrieve), ("advisor_reflect", adv.advisor_reflect),
                     ("advisor_answer", adv.advisor_answer), ("placement_lookup", adv.placement_lookup),
                     ("hitl_gate", hitl_gate), ("placement_finalize", placement_finalize),
                     ("assessor_execute", ass.assessor_execute), ("assessor_grade", ass.assessor_grade),
                     ("assessor_judge", ass.assessor_judge), ("assessor_unverified", ass.assessor_unverified),
                     ("assessor_finalize", ass.assessor_finalize)]:
        g.add_node(name, fn)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_by_intent)
    # Agentic RAG loop
    g.add_edge("advisor_retrieve", "advisor_reflect")
    g.add_conditional_edges("advisor_reflect", adv.route_after_reflect)
    g.add_edge("advisor_answer", END)
    # Evaluator-Optimizer loop
    g.add_conditional_edges("assessor_execute", ass.route_after_execute)
    g.add_edge("assessor_grade", "assessor_judge")
    g.add_conditional_edges("assessor_judge", ass.route_after_judge)
    g.add_edge("assessor_unverified", END)
    g.add_edge("assessor_finalize", END)
    # Placement + HITL
    g.add_conditional_edges("placement_lookup", adv.route_after_placement)
    g.add_edge("hitl_gate", "placement_finalize")
    g.add_edge("placement_finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def thread_config(learner_id: str) -> dict:
    """One thread per learner => isolated, persistent conversational state."""
    return {"configurable": {"thread_id": f"learner-{learner_id}"}}


# Per-turn fields. The checkpointer persists the WHOLE state per learner thread, so anything
# not reset here would leak into the next question (e.g. old code making a placement
# question look like a code submission, or a stale HITL approval being reused).
# `messages` is intentionally NOT reset: it is the multi-turn conversation memory.
_TURN_DEFAULTS = {
    "code": "", "assignment": "", "expected_output": "", "execution": {}, "grading": {},
    "judge": {}, "judge_feedback": "", "refinement_count": 0, "placement": {}, "hitl": None,
    "final_answer": "", "rag_passages": [], "rag_query": "", "rag_attempts": 0,
    "context_sufficient": False,
}


def ask(graph, learner_id: str, text: str, **extra):
    """Send a learner message into their own thread (memory kept, per-turn fields reset)."""
    payload = {**_TURN_DEFAULTS, "messages": [HumanMessage(text)], "learner_id": learner_id, **extra}
    return graph.invoke(payload, thread_config(learner_id))


def resume_hitl(graph, learner_id: str, approved: bool, reviewer: str, note: str = ""):
    """Human decision resumes the paused graph."""
    return graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer, "note": note}),
                        thread_config(learner_id))
