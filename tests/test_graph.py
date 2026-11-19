"""Graph-level tests with a FAKE LLM and fake retriever: no API key, no network.

They run the REAL LangGraph orchestrator (routing, loops, checkpointing, HITL) and replace only
the model/vector-store calls, so they prove control flow rather than model quality.
Run: python -m unittest tests.test_graph -v   (skipped automatically if langgraph is missing)
"""
import importlib.util
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

HAS_LG = importlib.util.find_spec("langgraph") is not None
FENCED = "Please grade:\n```python\nprint('hi')\n```"


@unittest.skipUnless(HAS_LG, "langgraph not installed")
class GraphTests(unittest.TestCase):
    def setUp(self):
        from src.agents import orchestrator as orch
        from src.schemas import AssessmentResult, ContextGrade, JudgeVerdict
        from src.tools.db_queries import build_database

        build_database()
        self.orch, self.calls = orch, {"retrieve": [], "grade": 0, "judge": 0}
        self.judge_approves = True
        self.retrieval_good_when = "checkpointer"       # reformulated query must contain this
        self.reflect_always_insufficient = False

        def fake_retrieve(query, k=4):
            self.calls["retrieve"].append(query)
            good = self.retrieval_good_when in query
            return {"status": "ok", "latency_ms": 1.0, "within_sla": True,
                    "passages": [{"source": "x.md", "text": "GOOD_CONTEXT" if good else "unrelated"}]}

        def fake_structured(schema, prompt, llm=None):
            if schema is ContextGrade:
                if "GOOD_CONTEXT" in prompt and not self.reflect_always_insufficient:
                    return ContextGrade(sufficient=True)
                return ContextGrade(sufficient=False, reformulated_query="checkpointer thread_id")
            if schema is AssessmentResult:
                self.calls["grade"] += 1
                return AssessmentResult(score=85, feedback_points=["ok"], pass_status="pass")
            if schema is JudgeVerdict:
                self.calls["judge"] += 1
                return JudgeVerdict(approved=self.judge_approves, issues=[] if self.judge_approves else ["nope"])
            raise AssertionError(schema)

        self.stack = ExitStack()
        for target, new in [
            ("src.agents.advisor_agent.retrieve_curriculum", fake_retrieve),
            ("src.agents.advisor_agent.invoke_structured", fake_structured),
            ("src.agents.advisor_agent.get_llm", lambda: SimpleNamespace(invoke=lambda p: SimpleNamespace(content="FAKE ANSWER"))),
            ("src.agents.assessor_agent.invoke_structured", fake_structured),
            ("src.agents.orchestrator.invoke_structured", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no LLM"))),
        ]:
            self.stack.enter_context(patch(target, new))
        self.addCleanup(self.stack.close)
        self.graph = orch.build_graph()

    # ---- Agentic RAG -------------------------------------------------------------------
    def test_rag_reformulates_then_answers(self):
        out = self.orch.ask(self.graph, "SPR-1001", "How is chat state kept per student?")
        self.assertEqual(len(self.calls["retrieve"]), 2)
        self.assertIn("checkpointer", self.calls["retrieve"][1])
        self.assertEqual(out["final_answer"], "FAKE ANSWER")

    def test_rag_gives_up_without_guessing(self):
        self.retrieval_good_when = "never-matches"
        out = self.orch.ask(self.graph, "SPR-1001", "Something not in the curriculum")
        self.assertEqual(len(self.calls["retrieve"]), 3)          # 1 search + 2 reformulations
        self.assertIn("couldn't find", out["final_answer"])

    # ---- Evaluator-Optimizer -----------------------------------------------------------
    def test_refinement_cap_is_exactly_three(self):
        self.judge_approves = False
        out = self.orch.ask(self.graph, "SPR-1002", FENCED)
        self.assertEqual(out["refinement_count"], 3)
        self.assertEqual(self.calls["grade"], 4)                   # 1 initial + 3 refinements
        self.assertEqual(self.calls["judge"], 4)
        self.assertIn("flagged for human review", out["final_answer"])

    def test_judge_approval_stops_immediately(self):
        out = self.orch.ask(self.graph, "SPR-1002", FENCED)
        self.assertEqual((self.calls["grade"], out["refinement_count"]), (1, 0))

    def test_sandbox_failure_returns_unable_to_verify(self):
        broken = {"status": "unable_to_verify", "stdout": "", "stderr": "boom"}
        with patch("src.agents.assessor_agent.execute_python", return_value=broken):
            out = self.orch.ask(self.graph, "SPR-1002", FENCED)
        self.assertIn("Unable to verify", out["final_answer"])
        self.assertEqual(self.calls["grade"], 0)                   # never graded blind

    # ---- Placement, HITL, memory -------------------------------------------------------
    def _paused(self, learner):
        return self.graph.get_state(self.orch.thread_config(learner)).next == ("hitl_gate",)

    def test_hitl_pauses_then_approve_labels_fawry(self):
        self.orch.ask(self.graph, "SPR-1001", "Which job openings match me?")
        self.assertTrue(self._paused("SPR-1001"))
        out = self.orch.resume_hitl(self.graph, "SPR-1001", True, "reviewer")
        self.assertIn("[approved by human reviewer]", out["final_answer"])

    def test_hitl_reject_removes_tier1(self):
        self.orch.ask(self.graph, "SPR-1001", "Which job openings match me?")
        out = self.orch.resume_hitl(self.graph, "SPR-1001", False, "reviewer")
        self.assertNotIn("Fawry", out["final_answer"])
        self.assertIn("Swvl", out["final_answer"])

    def test_ineligible_learner_never_pauses_or_gets_match(self):
        out = self.orch.ask(self.graph, "SPR-1021", "Any job openings for me?")
        self.assertFalse(self._paused("SPR-1021"))
        self.assertNotIn("Verified matches", out["final_answer"])

    def test_old_code_does_not_hijack_next_turn(self):
        """Regression: a placement question after a grading request must still route to placement."""
        self.orch.ask(self.graph, "SPR-1001", FENCED)
        out = self.orch.ask(self.graph, "SPR-1001", "Which job openings match me?")
        self.assertEqual(out["intent"], "placement_request")
        self.assertEqual(out["code"], "")

    def test_threads_are_isolated_per_learner(self):
        self.orch.ask(self.graph, "SPR-1001", "alpha question about checkpointer")
        self.orch.ask(self.graph, "SPR-1002", "beta question about checkpointer")
        texts = lambda lid: " ".join(str(m.content) for m in
                                     self.graph.get_state(self.orch.thread_config(lid)).values["messages"])
        self.assertIn("alpha", texts("SPR-1001"))
        self.assertNotIn("beta", texts("SPR-1001"))

    def test_memory_accumulates_within_a_thread(self):
        self.orch.ask(self.graph, "SPR-1001", "first question about checkpointer")
        self.orch.ask(self.graph, "SPR-1001", "second question about checkpointer")
        msgs = self.graph.get_state(self.orch.thread_config("SPR-1001")).values["messages"]
        joined = " ".join(str(m.content) for m in msgs)
        self.assertIn("first question", joined)
        self.assertIn("second question", joined)


if __name__ == "__main__":
    unittest.main()
