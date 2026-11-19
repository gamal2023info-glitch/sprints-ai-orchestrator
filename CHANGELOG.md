# Decision log
- D1 Matching is rule-based, not LLM-based (client's hallucination fear).
- D2 Placement text is templated from DB rows; LLM is used only for routing, RAG answers and grading.
- D3 Sandbox = subprocess + AST screen + rlimits (not in-process PythonREPL).
- D4 One flat StateGraph (no sub-graphs) so a single checkpointer/thread isolates each learner.
- D5 Refinement counter increments each time a grade is re-done after a judge rejection; cap = 3 refinements (4 grading rounds max). Earlier build stopped after 2 - fixed and covered by tests/test_graph.py.
- D6 Deterministic hard rules run after the LLM judge and cannot be overridden by it.
- D7 Heuristic router fallback if the LLM router fails.
