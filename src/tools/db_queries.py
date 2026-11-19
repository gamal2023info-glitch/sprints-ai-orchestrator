"""SQLite tools over the Sprints learner database (Sprint 1: ReAct agent + custom SQL tool).

Curriculum mapping: "Data & Tools" module - parameterised SQL, read-only access,
and grounding agent answers in live records instead of model memory.

Anti-hallucination design (client's #1 concern): every fact about a learner or a
partner opening comes from these functions. LLMs never invent qualifications;
``match_partners`` is deterministic rule-based code.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from typing import Any

from src.config import (CSV_PATH, DB_PATH, ELIGIBLE_STATUS, MIN_COMPLETED_MODULES,
                        TIER1_PARTNERS, UNABLE_TO_VERIFY)
from src.tools._tooling import lc_tool

LEARNER_ID_RE = re.compile(r"^SPR-\d{4}$")

# ASSUMPTION (flagged in docs/project_charter.md): the brief supplies no openings
# data, so these are illustrative partner openings used to demo live matching.
PARTNER_OPENINGS = [
    ("OPN-FAW-01", "Fawry", "AI Engineer (Agentic)", "Advanced Agentic AI", 25000, 45000, 85),
    ("OPN-FAW-02", "Fawry", "Backend Engineer", "Backend", 15000, 22000, 75),
    ("OPN-FAW-03", "Fawry", "Frontend Engineer", "Frontend", 12000, 18000, 75),
    ("OPN-FAW-04", "Fawry", "Data Scientist", "Data Science", 22000, 32000, 80),
    ("OPN-EAND-01", "e& Egypt", "AI Engineer (Agentic)", "Advanced Agentic AI", 25000, 40000, 80),
    ("OPN-EAND-02", "e& Egypt", "Backend Engineer", "Backend", 16000, 24000, 75),
    ("OPN-EAND-03", "e& Egypt", "Data Scientist", "Data Science", 22000, 32000, 80),
    ("OPN-EAND-04", "e& Egypt", "Frontend Engineer", "Frontend", 12000, 18000, 75),
    ("OPN-SWVL-01", "Swvl", "Frontend Engineer", "Frontend", 12000, 18000, 75),
    ("OPN-SWVL-02", "Swvl", "Data Scientist", "Data Science", 24000, 32000, 80),
    ("OPN-SWVL-03", "Swvl", "AI Engineer (Agentic)", "Advanced Agentic AI", 22000, 35000, 80),
    ("OPN-VOD-01", "Vodafone Egypt", "Backend Engineer", "Backend", 15000, 22000, 75),
    ("OPN-VOD-02", "Vodafone Egypt", "Data Scientist", "Data Science", 20000, 30000, 78),
]


def build_database(csv_path=CSV_PATH, db_path=DB_PATH) -> int:
    """(Re)build the SQLite DB from the learner CSV; returns the learner row count."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            DROP TABLE IF EXISTS learners; DROP TABLE IF EXISTS partner_openings;
            CREATE TABLE learners(
              learner_id TEXT PRIMARY KEY, name TEXT, bootcamp_track TEXT,
              completed_modules INTEGER, python_assignment_score INTEGER,
              target_salary_egp INTEGER, preferred_employers TEXT, readiness_status TEXT);
            CREATE TABLE partner_openings(
              opening_id TEXT PRIMARY KEY, employer TEXT, title TEXT, track TEXT,
              salary_min_egp INTEGER, salary_max_egp INTEGER, min_score INTEGER);
            """
        )
        con.executemany(
            "INSERT INTO learners VALUES (:learner_id,:name,:bootcamp_track,:completed_modules,"
            ":python_assignment_score,:target_salary_egp,:preferred_employers,:readiness_status)",
            rows,
        )
        con.executemany("INSERT INTO partner_openings VALUES (?,?,?,?,?,?,?)", PARTNER_OPENINGS)
        con.commit()
    finally:
        con.close()
    return len(rows)


def _connect_ro(db_path=DB_PATH) -> sqlite3.Connection:
    """Open the DB read-only so agent-written SQL can never mutate records."""
    if not db_path.exists():
        build_database(db_path=db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def get_learner(learner_id: str, db_path=DB_PATH) -> dict[str, Any]:
    """Fetch one learner (completed modules, EGP target salary, preferred employers)."""
    if not LEARNER_ID_RE.match(learner_id or ""):
        return {"status": "not_found", "message": f"Invalid learner id: {learner_id!r}"}
    try:
        con = _connect_ro(db_path)
        try:
            row = con.execute("SELECT * FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
        finally:
            con.close()
    except Exception as exc:  # graceful fallback required by the brief
        return {**UNABLE_TO_VERIFY, "error": type(exc).__name__}
    if row is None:
        return {"status": "not_found", "message": f"No learner {learner_id} in the database."}
    data = dict(row)
    data["preferred_employers"] = [e.strip() for e in data["preferred_employers"].split(",") if e.strip()]
    return {"status": "ok", "learner": data}


def match_partners(learner_id: str, db_path=DB_PATH) -> dict[str, Any]:
    """Deterministically match a learner to live partner openings.

    A learner is matched to an opening only if ALL hold: readiness is
    'Ready for Placement', completed_modules >= policy minimum, the employer is in
    the learner's own preferred list, the opening's track equals the learner's
    track, score >= the opening's min_score and target salary <= opening max.
    Each recommendation lists the reasons, so a human can audit it.
    """
    res = get_learner(learner_id, db_path)
    if res["status"] != "ok":
        return res
    p = res["learner"]
    try:
        con = _connect_ro(db_path)
        try:
            openings = [dict(r) for r in con.execute("SELECT * FROM partner_openings")]
        finally:
            con.close()
    except Exception as exc:
        return {**UNABLE_TO_VERIFY, "error": type(exc).__name__}

    if p["readiness_status"] != ELIGIBLE_STATUS:
        return {"status": "not_eligible", "learner": p, "recommendations": [],
                "message": f"Readiness is '{p['readiness_status']}', not '{ELIGIBLE_STATUS}'."}
    if p["completed_modules"] < MIN_COMPLETED_MODULES:
        return {"status": "not_eligible", "learner": p, "recommendations": [],
                "message": f"Only {p['completed_modules']} modules completed (< {MIN_COMPLETED_MODULES})."}

    recs, rejected = [], []
    for o in openings:
        why_not = None
        if o["employer"] not in p["preferred_employers"]:
            why_not = "employer not in learner's preferences"
        elif o["track"] != p["bootcamp_track"]:
            why_not = f"track mismatch ({o['track']} vs {p['bootcamp_track']})"
        elif p["python_assignment_score"] < o["min_score"]:
            why_not = f"score {p['python_assignment_score']} < required {o['min_score']}"
        elif p["target_salary_egp"] > o["salary_max_egp"]:
            why_not = f"target {p['target_salary_egp']} EGP > band max {o['salary_max_egp']}"
        if why_not:
            if o["employer"] in p["preferred_employers"]:
                rejected.append({"opening_id": o["opening_id"], "reason": why_not})
            continue
        recs.append({**o, "tier1": o["employer"] in TIER1_PARTNERS,
                     "reasons": [f"readiness={p['readiness_status']}",
                                 f"track={p['bootcamp_track']}",
                                 f"score={p['python_assignment_score']}>={o['min_score']}",
                                 f"target={p['target_salary_egp']} EGP within band "
                                 f"{o['salary_min_egp']}-{o['salary_max_egp']}"]})
    return {"status": "ok" if recs else "no_match", "learner": p,
            "recommendations": recs, "rejected": rejected}


_FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


@lc_tool
def query_learner_db(sql: str) -> str:
    """Run ONE read-only SELECT against the Sprints learner DB (tables: learners, partner_openings).

    Use it to read a learner's completed_modules, target_salary_egp (EGP) and
    preferred_employers. Max 50 rows. Returns JSON text, or a structured
    'Unable to verify' message if the database fails.
    """
    import json

    stmt = (sql or "").strip().rstrip(";")
    if not stmt.lower().startswith("select") or ";" in stmt or _FORBIDDEN_SQL.search(stmt):
        return json.dumps({"status": "rejected", "message": "Only a single SELECT is allowed."})
    try:
        con = _connect_ro()
        try:
            rows = [dict(r) for r in con.execute(stmt).fetchmany(50)]
        finally:
            con.close()
        return json.dumps({"status": "ok", "rows": rows}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({**UNABLE_TO_VERIFY, "error": type(exc).__name__})


def build_react_sql_agent():
    """Sprint 1 deliverable: a simple ReAct agent equipped with the SQL tool."""
    from langgraph.prebuilt import create_react_agent

    from src.llm import get_llm

    return create_react_agent(
        get_llm(), [query_learner_db],
        prompt=("You answer questions about Sprints learners using ONLY the query_learner_db tool. "
                "Never state a fact you did not read from the tool output. "
                "If the tool fails, reply exactly: Unable to verify."),
    )
