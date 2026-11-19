"""Central configuration for the Sprints AI Career & Assessment Orchestrator.

Every constant maps to a number in the client brief / discovery call so a
reviewer can trace a requirement to the single place where it is enforced.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "sprints_learner_db.csv"
DB_PATH = DATA_DIR / "sprints_learner.db"
CURRICULUM_DIR = DATA_DIR / "curriculum_docs"
CHROMA_DIR = DATA_DIR / "chroma_store"

# --- RAG (Sprint 1): 500-token chunks, 50-token overlap, <1000 ms retrieval ---
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
RETRIEVAL_SLA_MS = 1000
RAG_TOP_K = 4
MAX_RAG_REFORMULATIONS = 2

# --- Evaluator-Optimizer: strictly <= 3 refinement iterations ---
MAX_REFINEMENTS = 3
PASS_SCORE = 70

# --- HITL: placements to these partners need human approval ---
TIER1_PARTNERS = frozenset({"Fawry"})

# --- Placement eligibility policy (ASSUMPTION: confirm with Sprints) ---
MIN_COMPLETED_MODULES = 10
ELIGIBLE_STATUS = "Ready for Placement"

# --- Sandbox limits (Python REPL tool) ---
SANDBOX_TIMEOUT_S = 5
SANDBOX_MEMORY_MB = 256
SANDBOX_MAX_OUTPUT_CHARS = 4000

# Groq serves chat models only (no embeddings API) -> embeddings run locally via fastembed.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

UNABLE_TO_VERIFY = {
    "status": "unable_to_verify",
    "message": "Unable to verify: a required tool failed. No answer was guessed.",
}
