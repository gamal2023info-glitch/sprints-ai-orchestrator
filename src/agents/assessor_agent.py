"""Assessor Agent: sandboxed grading with an Evaluator-Optimizer loop (Sprint 2).

Curriculum mapping: "Evaluator-Optimizer pattern". Flow:
execute (sandbox) -> grade (Pydantic JSON) -> judge (LLM-as-a-Judge vs rubric)
-> approved? finish : refine (max MAX_REFINEMENTS times) -> finish.
"""
from __future__ import annotations

from src.agents.messages import ai_msg, tool_msg
from src.config import MAX_REFINEMENTS, PASS_SCORE, UNABLE_TO_VERIFY
from src.llm import invoke_structured
from src.schemas import (ASSESSOR_PROMPT, JUDGE_PROMPT, RUBRIC, AssessmentResult, JudgeVerdict)
from src.state import OrchestratorState
from src.tools.code_repl import execute_python

_RUBRIC = RUBRIC.format(pass_score=PASS_SCORE)


def assessor_execute(state: OrchestratorState) -> dict:
    """Run the learner's code in the sandbox; evidence for all later grading."""
    result = execute_python(state["code"])
    return {"execution": result, "refinement_count": 0, "judge": {}, "judge_feedback": "",
            "messages": [tool_msg("python_sandbox", result)]}


def route_after_execute(state: OrchestratorState) -> str:
    """If the sandbox itself failed, return 'Unable to verify' instead of guessing a grade."""
    return "assessor_unverified" if state["execution"]["status"] == "unable_to_verify" else "assessor_grade"


def assessor_unverified(state: OrchestratorState) -> dict:
    """Graceful fallback (acceptance criterion): structured 'Unable to verify'."""
    msg = UNABLE_TO_VERIFY["message"]
    return {"grading": dict(UNABLE_TO_VERIFY), "final_answer": msg, "messages": [ai_msg("assessor", msg)]}


def assessor_grade(state: OrchestratorState) -> dict:
    """Produce (or refine) a grade as a validated AssessmentResult."""
    ex = state["execution"]
    prompt = ASSESSOR_PROMPT.format(
        rubric=_RUBRIC, assignment=state.get("assignment", "n/a"),
        expected=state.get("expected_output", ""), code=state["code"], status=ex["status"],
        stdout=ex["stdout"], stderr=ex["stderr"], judge_feedback=state.get("judge_feedback", ""))
    grade: AssessmentResult = invoke_structured(AssessmentResult, prompt)
    # A grading call that follows a judge rejection IS a refinement; count it here so the
    # cap means exactly "at most 3 refinements" (1 initial grade + up to 3 refined grades).
    prev = state.get("judge") or {}
    is_refinement = bool(prev) and not prev.get("approved", True)
    count = state.get("refinement_count", 0) + (1 if is_refinement else 0)
    return {"grading": grade.model_dump(), "refinement_count": count,
            "messages": [ai_msg("assessor", f"Draft grade: {grade.model_dump_json()}")]}


def _hard_rules(state: OrchestratorState) -> list[str]:
    """Deterministic checks that no LLM judge may override."""
    g, ex, issues = state["grading"], state["execution"], []
    if ex["status"] != "ok" and g["pass_status"] == "pass":
        issues.append(f"Sandbox status is {ex['status']}; this can never be a pass.")
    if (g["score"] >= PASS_SCORE) != (g["pass_status"] == "pass"):
        issues.append(f"pass_status disagrees with score vs threshold {PASS_SCORE}.")
    exp = state.get("expected_output", "").strip()
    if exp and ex["status"] == "ok" and ex["stdout"].strip() != exp and g["pass_status"] == "pass":
        issues.append("Output differs from expected output but graded pass.")
    return issues


def assessor_judge(state: OrchestratorState) -> dict:
    """LLM-as-a-Judge reviews the draft grade against the rubric + hard rules."""
    ex = state["execution"]
    verdict: JudgeVerdict = invoke_structured(JudgeVerdict,
        JUDGE_PROMPT.format(pass_score=PASS_SCORE, rubric=_RUBRIC, status=ex["status"],
                            stdout=ex["stdout"], stderr=ex["stderr"], grade=state["grading"]))
    issues = list(verdict.issues) + _hard_rules(state)
    approved = verdict.approved and not issues
    return {"judge": {"approved": approved, "issues": issues},
            "judge_feedback": "; ".join(issues),
            "messages": [ai_msg("judge", f"approved={approved} issues={issues}")]}


def route_after_judge(state: OrchestratorState) -> str:
    """Halt when approved OR after MAX_REFINEMENTS (3) refinement rounds."""
    if state["judge"]["approved"] or state["refinement_count"] >= MAX_REFINEMENTS:
        return "assessor_finalize"
    return "assessor_grade"


def assessor_finalize(state: OrchestratorState) -> dict:
    """Return the grade; flag it if the loop ended without judge approval."""
    g = dict(state["grading"])
    note = "" if state["judge"]["approved"] else (
        f"\n(Note: judge did not fully approve after {state['refinement_count']} refinements; "
        f"flagged for human review: {state['judge']['issues']})")
    text = (f"Score: {g['score']}/100 - {g['pass_status'].upper()}\n"
            + "\n".join(f"- {p}" for p in g["feedback_points"]) + note)
    return {"final_answer": text, "messages": [ai_msg("assessor", text)]}
