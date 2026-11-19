# Sprints AI Orchestrator

LangGraph multi-agent system: curriculum Q&A (Agentic RAG), sandboxed Python grading
(Evaluator-Optimizer) and DB-grounded partner matching with Human-in-the-Loop.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Windows PowerShell: .venv\Scripts\Activate.ps1
cp .env.example .env   # then put your GROQ_API_KEY inside
python -m src.list_models         # shows which Groq models your key can use
python -m src.main --setup        # SQLite from CSV + Chroma ingestion
python -m src.main --demo -v      # live demo (-v shows Human/AI/Tool messages)
python -m unittest discover -s tests -v   # no API key needed (graph tests use a fake LLM)
python -m src.main --react        # ReAct + SQL agent scene
python -m src.main --rag-loop     # RAG self-reflection / re-search scene
```

## Layout
`src/state.py` global TypedDict - `src/agents/` orchestrator + workers - `src/tools/` RAG, sandbox, SQL -
`notebooks/` evaluator-optimizer demo - `docs/project_charter.md` assumptions & risks.

## Requirement map
| Requirement | Where |
|---|---|
| Ingestion 500/50 tokens -> Chroma | `tools/rag_retriever.py` |
| Pydantic JSON (score, feedback_points, pass_status) | `schemas.py` |
| ReAct + SQL tool | `tools/db_queries.py::build_react_sql_agent` |
| Python REPL sandbox | `tools/code_repl.py` |
| StateGraph + TypedDict routing | `state.py`, `agents/orchestrator.py` |
| MemorySaver per learner thread | `orchestrator.thread_config` |
| Agentic RAG self-reflection | `advisor_agent.py` |
| Evaluator-Optimizer (max 3) | `assessor_agent.py` |
| HITL interrupt (Fawry) | `orchestrator.hitl_gate` |
| Message protocol + logs | `agents/messages.py` |
| Graceful "Unable to verify" | `db_queries.py`, `code_repl.py`, `assessor_unverified` |

## Status
Tools verified by unit tests. The LangGraph/Groq flow was written but **not executed** in the
authoring environment (no network/API key): run the demo once and record the evidence.
