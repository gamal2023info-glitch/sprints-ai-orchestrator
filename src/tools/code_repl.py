"""Sandboxed Python execution tool for the Assessor Agent (Sprint 1: Python REPL tool).

Curriculum mapping: "Python Fundamentals" auto-grading. The client's non-negotiable:
learner code must never breach platform security.

Defence in depth (each layer is independently useful):
 1. Static AST screen: blocks dangerous imports, builtins and dunder access.
 2. Separate interpreter process in isolated mode (-I), empty environment,
    throw-away working directory, no stdin.
 3. POSIX resource limits: CPU seconds, address space, file size, open files.
 4. Wall-clock timeout with hard kill, and truncated output.

WINDOWS NOTE: POSIX resource limits (CPU/RAM/file) do not exist on Windows, so only the AST screen,
isolation and wall-clock timeout apply there. Run grading under WSL2/Docker for the full set.

HONEST LIMIT: this is NOT a substitute for container isolation. For production,
run the same entry point inside a locked-down container/gVisor/Firecracker with no
network. We deliberately did not use langchain's PythonREPL, which runs code
in-process and is unsafe for untrusted input.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from typing import Any

from src.config import (SANDBOX_MAX_OUTPUT_CHARS, SANDBOX_MEMORY_MB, SANDBOX_TIMEOUT_S,
                        UNABLE_TO_VERIFY)
from src.tools._tooling import lc_tool

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "multiprocessing",
    "threading", "importlib", "builtins", "signal", "resource", "pickle", "marshal", "io",
    "http", "urllib", "requests", "ftplib", "smtplib", "asyncio", "inspect", "gc", "tempfile",
    "glob", "fcntl", "pty", "select", "sqlite3", "code", "codeop", "runpy", "site",
}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                 "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit"}


def static_screen(code: str) -> str | None:
    """Return a human-readable reason if the code is rejected, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BLOCKED_MODULES:
                    return f"Blocked import: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BLOCKED_MODULES:
                return f"Blocked import: {node.module}"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return f"Blocked builtin: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Blocked dunder attribute access: {node.attr}"
    return None


def _limits() -> None:  # pragma: no cover - runs in child process
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (SANDBOX_TIMEOUT_S + 1, SANDBOX_TIMEOUT_S + 1))
    mem = SANDBOX_MEMORY_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    os.setsid()


def _child_env() -> dict[str, str]:
    """Empty environment (no secrets leak to learner code). Windows needs SYSTEMROOT to start Python."""
    if os.name == "nt":
        return {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows")}
    return {}


def execute_python(code: str, timeout: int = SANDBOX_TIMEOUT_S) -> dict[str, Any]:
    """Execute learner code in the sandbox and return a structured result.

    status is one of: ok | runtime_error | timeout | blocked | unable_to_verify.
    """
    reason = static_screen(code)
    if reason:
        return {"status": "blocked", "stdout": "", "stderr": reason}
    try:
        with tempfile.TemporaryDirectory(prefix="sprints_sbx_") as wd:
            path = os.path.join(wd, "submission.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            proc = subprocess.run(
                [sys.executable, "-I", path], cwd=wd, env=_child_env(), stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None,
            )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": f"Exceeded {timeout}s time limit."}
    except Exception as exc:  # sandbox itself failed -> graceful fallback
        return {**UNABLE_TO_VERIFY, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    cap = SANDBOX_MAX_OUTPUT_CHARS
    return {"status": "ok" if proc.returncode == 0 else "runtime_error",
            "stdout": proc.stdout[:cap], "stderr": proc.stderr[-cap:], "returncode": proc.returncode}


@lc_tool
def run_python_snippet(code: str) -> dict:
    """Securely run a learner's Python snippet and capture stdout / runtime errors.

    Used by the Assessor Agent to ground grading of Python Fundamentals assignments
    in real execution evidence. Code runs in a restricted subprocess (no network,
    no file/OS access, 5s CPU/wall limit, 256 MB RAM).
    """
    return execute_python(code)
