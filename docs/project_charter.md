# Project Charter - Sprints Multi-Agent AI Career & Assessment Orchestrator

**Problem.** Student-success staff are overwhelmed by repetitive curriculum questions, manual Python
grading and hand-matching graduates to partners (Fawry, e& Egypt, Swvl). Target: cut learner wait time by 80%.

**Scope.** One LangGraph StateGraph: Orchestrator -> Advisor (Agentic RAG, placement) and Assessor
(sandboxed grading + Evaluator-Optimizer); MemorySaver per learner; HITL interrupt for tier-1 (Fawry).

**Success criteria (from the discovery call).** Live demo of: (1) curriculum answer, (2) secure grading,
(3) accurate routing to a partner opening from live DB records. Zero invented qualifications.

**Non-negotiables.** Unified LangGraph + TypedDict + checkpointing; strict sandbox for learner code.

**Risks & mitigations.**
| Risk | Mitigation |
|---|---|
| Hallucinated career matches | Matching is deterministic SQL/rules; reply is a template; LLM never writes employer facts |
| Sandbox escape | AST screen + isolated subprocess + rlimits; production must add container isolation |
| Evaluator loop cost | Hard cap of 3 refinements, deterministic rules the judge cannot override |
| Retrieval > 1000 ms | Cached Chroma handle, k=4, latency measured and reported per call |

**Assumptions / data gaps (confirm with Youssef).**
1. The deck says 500 learners; the supplied CSV has **25**.
2. No curriculum documents were supplied: `data/curriculum_docs/` holds PLACEHOLDER files.
3. No partner-openings data was supplied: `partner_openings` is illustrative mock data.
4. Eligibility policy (>=10 modules, status "Ready for Placement", score >= opening minimum,
   target salary <= band max) is our assumption.
5. "Tier-1" = Fawry only (the sole example in the brief).
6. The call mentions Vodafone only via the CSV; it is treated as a normal (non tier-1) partner.
7. The audio transcript says "Sprintal" and "TypeState"; interpreted as Sprints and TypedDict.
8. The brief says "Python REPL utility"; we used a hardened subprocess because the in-process
   REPL cannot satisfy the "strictly sandboxed" constraint.
