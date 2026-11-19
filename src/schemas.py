"""Pydantic schemas + prompt templates (Sprint 1: typed prompts & output parsers).

The Assessor's JSON contract is exactly: score, feedback_points, pass_status.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AssessmentResult(BaseModel):
    """Structured grade for a learner's Python submission (Python Fundamentals module)."""

    score: int = Field(ge=0, le=100, description="0-100 grade against the rubric")
    feedback_points: list[str] = Field(min_length=1, description="Concrete, actionable points")
    pass_status: Literal["pass", "fail"]

    @field_validator("feedback_points")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p.strip()] or ["No feedback produced."]


class JudgeVerdict(BaseModel):
    """'LLM as a Judge' verdict on the Assessor's grade."""

    approved: bool
    issues: list[str] = Field(default_factory=list, description="What must change if not approved")


class RouteDecision(BaseModel):
    """Intent routing decision made by the Orchestrator."""

    intent: Literal["curriculum_question", "code_submission", "placement_request"]


class ContextGrade(BaseModel):
    """Self-reflection result for Agentic RAG."""

    sufficient: bool
    reformulated_query: str = Field(default="", description="Better search query if insufficient")


RUBRIC = """Grading rubric (100 pts):
1. Correctness (50): does the code run and produce the expected behaviour/output?
2. Code quality (25): readable names, no dead code, idiomatic Python.
3. Robustness (15): handles edge cases / no runtime errors.
4. Docstrings & clarity (10).
Hard rules: a runtime error or timeout can NEVER be 'pass'. Pass threshold = {pass_score}."""

ASSESSOR_PROMPT = """You are the Sprints Assessor Agent (Python Fundamentals curriculum).
Grade ONLY from the evidence below. Do not assume behaviour you did not see.

{rubric}

Assignment: {assignment}
Expected output (may be empty): {expected}

Submitted code:
```python
{code}
```
Sandbox result: status={status}
stdout:
{stdout}
stderr:
{stderr}

Reviewer feedback to address (empty on first pass): {judge_feedback}
"""

JUDGE_PROMPT = """You are an independent grader auditing another grader.
Check the proposed grade against the rubric AND the sandbox evidence.
Reject if: the score contradicts the evidence, a runtime error was marked pass,
feedback is vague, or pass_status disagrees with the threshold ({pass_score}).

{rubric}

Sandbox status={status}
stdout: {stdout}
stderr: {stderr}
Proposed grade: {grade}
"""

ROUTER_PROMPT = """Classify the learner's message into exactly one intent:
- curriculum_question: asks about course content/concepts
- code_submission: submits Python code to be graded
- placement_request: asks about job matching / partner openings / placement
Message: {message}"""

REFLECT_PROMPT = """Question: {question}
Retrieved curriculum excerpts:
{context}

Do the excerpts contain enough information to answer the question WITHOUT outside knowledge?
If not, propose a better search query."""

ANSWER_PROMPT = """You are the Sprints Advisor Agent. Answer ONLY from the curriculum excerpts.
If they do not contain the answer, say so. Cite the source file names.

Excerpts:
{context}

Question: {question}"""
