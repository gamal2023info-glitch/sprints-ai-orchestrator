"""Dependency-free tests for the tools that carry the client's hard constraints.

Run: python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest
from pathlib import Path

from src.tools import db_queries as db
from src.tools.code_repl import execute_python, static_screen


class TestDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp()) / "t.db"
        cls.n = db.build_database(db_path=cls.tmp)

    def test_row_count(self):
        self.assertEqual(self.n, 25)

    def test_profile_fields(self):
        r = db.get_learner("SPR-1001", self.tmp)
        self.assertEqual(r["learner"]["target_salary_egp"], 30000)
        self.assertEqual(r["learner"]["preferred_employers"], ["Fawry", "Swvl"])

    def test_unknown_and_malformed_ids_never_invented(self):
        self.assertEqual(db.get_learner("SPR-9999", self.tmp)["status"], "not_found")
        self.assertEqual(db.get_learner("1001; DROP TABLE learners", self.tmp)["status"], "not_found")

    def test_match_is_grounded_and_tier1_flagged(self):
        r = db.match_partners("SPR-1001", self.tmp)
        self.assertEqual(r["status"], "ok")
        employers = {x["employer"] for x in r["recommendations"]}
        self.assertTrue(employers <= {"Fawry", "Swvl"})          # only preferred employers
        self.assertTrue(any(x["tier1"] for x in r["recommendations"]))

    def test_not_ready_learner_gets_no_match(self):
        for lid in ("SPR-1021", "SPR-1007", "SPR-1002"):
            r = db.match_partners(lid, self.tmp)
            self.assertEqual(r["status"], "not_eligible", lid)
            self.assertEqual(r["recommendations"], [])

    def test_db_failure_returns_unable_to_verify(self):
        bad = Path(tempfile.mkdtemp()) / "x" / "missing.db"
        r = db.get_learner("SPR-1001", bad)
        self.assertEqual(r["status"], "unable_to_verify")

    def test_sql_tool_is_read_only(self):
        for bad in ("DELETE FROM learners", "SELECT 1; DROP TABLE learners", "UPDATE learners SET name='x'"):
            run = getattr(db.query_learner_db, "invoke", None)
            out = run({"sql": bad}) if run else db.query_learner_db(bad)
            self.assertIn("rejected", out)


class TestSandbox(unittest.TestCase):
    def test_ok_stdout(self):
        r = execute_python("print(sum(range(5)))")
        self.assertEqual((r["status"], r["stdout"].strip()), ("ok", "10"))

    def test_runtime_error_captured(self):
        r = execute_python("print(1/0)")
        self.assertEqual(r["status"], "runtime_error")
        self.assertIn("ZeroDivisionError", r["stderr"])

    def test_timeout(self):
        self.assertEqual(execute_python("while True: pass", timeout=2)["status"], "timeout")

    def test_blocked_constructs(self):
        for code in ["import os", "from subprocess import run", "open('/etc/passwd')",
                     "().__class__.__bases__", "eval('1')", "getattr(int,'x')", "__import__('os')"]:
            self.assertIsNotNone(static_screen(code), code)
            self.assertEqual(execute_python(code)["status"], "blocked", code)

    @unittest.skipUnless(os.name == "posix", "RAM rlimit is POSIX-only")
    def test_memory_bomb_contained(self):
        r = execute_python("x = 'a' * (10**10)\nprint(len(x))")
        self.assertNotEqual(r["status"], "ok")


if __name__ == "__main__":
    unittest.main()
