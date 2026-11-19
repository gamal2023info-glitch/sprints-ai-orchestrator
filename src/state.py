"""Global orchestrator state (Sprint 2: unified StateGraph with a TypedDict).

One state object flows through Orchestrator -> Advisor/Assessor workers. It is
checkpointed per learner thread (MemorySaver) so context never leaks between students.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["curriculum_question", "code_submission", "placement_request"]


class OrchestratorState(TypedDict, total=False):
    """Shared state. `messages` uses the add_messages reducer = persistent chat history."""

    messages: Annotated[list[AnyMessage], add_messages]
    learner_id: str
    intent: Intent
    # Advisor / Agentic RAG
    question: str
    rag_query: str
    rag_passages: list[dict]
    rag_attempts: int
    context_sufficient: bool
    # Assessor / Evaluator-Optimizer
    code: str
    assignment: str
    expected_output: str
    execution: dict
    grading: dict
    judge: dict
    judge_feedback: str
    refinement_count: int
    # Placement / HITL
    placement: dict
    hitl: dict
    final_answer: str
