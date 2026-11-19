"""Tiny shim so pure-Python tool logic stays importable/testable without LangChain."""
try:  # pragma: no cover
    from langchain_core.tools import tool as lc_tool
except ImportError:  # pragma: no cover
    def lc_tool(fn=None, **_kw):
        """Identity decorator used when langchain-core is not installed."""
        return fn if fn is not None else (lambda f: f)
