# SPDX-License-Identifier: Apache-2.0
"""Request-level tool authorization, independent of parser and tool choice."""

import logging

logger = logging.getLogger(__name__)


def declared_names(request) -> set[str]:
    original: set[str] | None = getattr(request, "_tool_authorization_names", None)
    if original is not None:
        return original
    if hasattr(request, "model_dump"):
        request = request.model_dump()
    names = set()
    for tool in (request or {}).get("tools") or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if isinstance(tool, dict):
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                names.add(function["name"])
    return names


def call_name(call):
    if hasattr(call, "model_dump"):
        call = call.model_dump()
    return call.get("function", call).get("name")


def log_drop(name):
    logger.warning("Dropping call to undeclared tool %r", name)
