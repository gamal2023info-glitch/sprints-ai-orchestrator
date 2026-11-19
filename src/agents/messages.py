"""Agentic Protocol helpers: uniform HumanMessage / AIMessage / ToolMessage traffic + verbose logs."""
from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import AIMessage, ToolMessage

log = logging.getLogger("sprints.protocol")


def ai_msg(agent: str, content: str) -> AIMessage:
    """AIMessage attributed to a named agent (orchestrator/advisor/assessor)."""
    log.info("[AIMessage] %s -> %s", agent, content[:160].replace("\n", " "))
    return AIMessage(content=content, name=agent)


def tool_msg(tool: str, payload: dict) -> ToolMessage:
    """ToolMessage wrapping a structured tool result (traceable in verbose logs)."""
    body = json.dumps(payload, ensure_ascii=False, default=str)
    log.info("[ToolMessage] %s <- %s", tool, body[:160])
    return ToolMessage(content=body, name=tool, tool_call_id=f"{tool}-{uuid.uuid4().hex[:8]}")
