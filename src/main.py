"""Live end-to-end demo (what the client asked for instead of slides).

    python -m src.main --setup     # build SQLite DB + ingest curriculum into Chroma
    python -m src.main --demo      # scene 1 curriculum Q&A, 2 secure grading, 3 placement + HITL
    python -m src.main --demo --auto-approve   # non-interactive (for recordings/CI)
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

SCENE_CODE = '''```python
def is_even(n):
    """Return True when n is even."""
    return n % 2 == 0

print(is_even(4), is_even(7))
```'''
EVIL_CODE = "```python\nimport os\nprint(os.listdir('/'))\n```"


class _Tee:
    """Write everything to the console AND to a UTF-8 log file (demo evidence for submission)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def banner(t: str) -> None:
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def run_react() -> None:
    """Sprint 1 deliverable: a plain ReAct agent that answers from the SQL tool only."""
    from src.tools.db_queries import build_react_sql_agent

    banner("Scene 5 - ReAct agent + custom SQL tool")
    out = build_react_sql_agent().invoke({"messages": [(
        "user", "For learner SPR-1001 give completed modules, target salary in EGP and preferred employers.")]})
    for m in out["messages"]:
        print(f"{type(m).__name__}: {getattr(m, 'tool_calls', None) or m.content}")


def run_rag_loop() -> None:
    """Show the Agentic RAG self-reflection loop on a question the curriculum cannot answer.

    Expected: retrieve -> reflect (insufficient) -> reformulate -> retrieve ... (max 2 reformulations)
    -> honest "couldn't find it" instead of an invented answer.
    """
    from src.agents.orchestrator import ask, build_graph
    from src.tools.rag_retriever import warm_up

    warm_up()
    banner("Scene 6 - Agentic RAG re-search on an out-of-curriculum question")
    out = ask(build_graph(), "SPR-1003", "What is the Sprints refund policy for withdrawn learners?")
    print("searches made:", out.get("rag_attempts"), "| last query:", out.get("rag_query"))
    print(out["final_answer"])


def run_demo(auto_approve: bool) -> None:
    from src.agents.orchestrator import ask, build_graph, resume_hitl, thread_config

    from src.tools.rag_retriever import warm_up

    warm_up()
    graph = build_graph()

    banner("Scene 1 - curriculum question (Agentic RAG)")
    out = ask(graph, "SPR-1001", "What is the difference between a list and a tuple in Python?")
    print(out["final_answer"])

    banner("Scene 2a - secure grading (Evaluator-Optimizer)")
    out = ask(graph, "SPR-1001", "Please grade my submission:\n" + SCENE_CODE,
              assignment="Write is_even(n) and print results for 4 and 7", expected_output="True False")
    print(out["final_answer"])

    banner("Scene 2b - malicious submission is blocked by the sandbox")
    out = ask(graph, "SPR-1002", "Grade this:\n" + EVIL_CODE, assignment="List files")
    print("sandbox:", out["execution"])

    banner("Scene 3 - placement from live DB records + HITL (Fawry is tier-1)")
    out = ask(graph, "SPR-1001", "Which partner openings match me for placement?")
    state = graph.get_state(thread_config("SPR-1001"))
    if state.next:  # graph is paused at hitl_gate
        payload = state.tasks[0].interrupts[0].value
        print("PAUSED for human approval:", payload["learner"]["name"],
              [f"{r['employer']} - {r['title']}" for r in payload["tier1_recommendations"]])
        ok = True if auto_approve else input("Approve? [y/N] ").strip().lower() == "y"
        out = resume_hitl(graph, "SPR-1001", ok, reviewer="Youssef El-Sherif")
    print(out["final_answer"])

    banner("Scene 4 - ineligible learner gets NO invented match")
    print(ask(graph, "SPR-1021", "Any job openings for me?")["final_answer"])


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--react", action="store_true", help="also run the ReAct + SQL agent scene")
    ap.add_argument("--rag-loop", action="store_true", help="also run the RAG re-search (self-reflection) scene")
    ap.add_argument("-v", "--verbose", action="store_true", help="log Agentic Protocol messages")
    ap.add_argument("--log-file", help="also save the full console output to this file (UTF-8)")
    a = ap.parse_args()
    # Windows consoles/pipes often default to cp1252/cp1256 and crash on characters models emit.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if a.log_file:
        log_fh = open(a.log_file, "w", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, log_fh), _Tee(sys.stderr, log_fh)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    if a.setup:
        from src.tools.db_queries import build_database
        from src.tools.rag_retriever import ingest_curriculum
        print("learners loaded:", build_database())
        print("chunks stored:", ingest_curriculum())
    if a.demo:
        run_demo(a.auto_approve)
    if a.react:
        run_react()
    if a.rag_loop:
        run_rag_loop()


if __name__ == "__main__":
    main()
