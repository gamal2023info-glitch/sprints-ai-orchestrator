"""LLM factory (single place to swap the model). Provider: Groq.

Groq retires/changes models often and access differs per account, so instead of
hard-coding one id we ask the API which models YOUR key can use and pick the first
preferred one that is available. Override with GROQ_MODEL in .env.
"""
from __future__ import annotations

import os
from functools import lru_cache

# Ordered by preference: strong tool-calling / structured-output models first.
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct",
    "llama-3.1-8b-instant",
]


def list_available_models() -> list[str]:
    """Return the chat-model ids the current GROQ_API_KEY can call."""
    from groq import Groq

    return sorted(m.id for m in Groq().models.list().data)


def pick_model() -> str:
    """GROQ_MODEL if set, else the first PREFERRED model your key can access."""
    forced = os.getenv("GROQ_MODEL", "").strip()
    if forced:
        return forced
    available = set(list_available_models())
    for m in PREFERRED_MODELS:
        if m in available:
            return m
    raise RuntimeError(
        "None of the preferred Groq models are available to your key. "
        "Run `python -m src.list_models` and set GROQ_MODEL in .env to one of them.")


@lru_cache(maxsize=1)
def get_llm():
    """Return the temperature-0 Groq chat model used by every agent.

    Temperature 0 is deliberate: the client's top fear is hallucination, so we
    remove sampling randomness wherever a decision feeds a business outcome.
    """
    from langchain_groq import ChatGroq

    model = pick_model()
    print(f"[llm] using Groq model: {model}")
    return ChatGroq(model=model, temperature=0, max_retries=3)


_WORKING_METHOD: dict[str, str] = {}   # schema name -> method that worked (saves wasted calls / rate limit)


def _extract_json(text: str) -> str:
    """Pull the outermost {...} object out of a model reply (handles ```json fences / prose)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in model reply: {text[:120]!r}")
    return text[start:end + 1]


def invoke_structured(schema, prompt: str, llm=None):
    """Call the LLM and return a validated `schema` (Pydantic) instance, whatever the model supports.

    Different Groq models support different structured-output mechanisms, and some
    reasoning models answer in prose instead of calling the schema tool. We try, in order:
      1. json_schema   (native structured outputs)
      2. function_calling (tool call)
      3. plain JSON: prompt for JSON only, extract it, validate with Pydantic
    The method that works is remembered per schema. Validation always happens in Pydantic,
    so malformed model output can never leak into the pipeline.
    """
    llm = llm or get_llm()
    name = schema.__name__
    order = ["json_schema", "function_calling", "plain_json"]
    if name in _WORKING_METHOD:
        order = [_WORKING_METHOD[name]] + [m for m in order if m != _WORKING_METHOD[name]]
    last_exc: Exception | None = None
    for method in order:
        try:
            if method == "plain_json":
                import json

                hint = ("\n\nRespond with ONLY a JSON object matching this JSON schema, no prose, "
                        f"no markdown:\n{json.dumps(schema.model_json_schema())}")
                raw = llm.invoke(prompt + hint).content
                result = schema.model_validate_json(_extract_json(raw))
            else:
                result = llm.with_structured_output(schema, method=method).invoke(prompt)
                if isinstance(result, dict):
                    result = schema.model_validate(result)
            _WORKING_METHOD[name] = method
            return result
        except Exception as exc:  # try next method
            last_exc = exc
    raise RuntimeError(f"Structured output failed for {name}: {last_exc}") from last_exc
