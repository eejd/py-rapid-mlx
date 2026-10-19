# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for MiniCPM5's documented native XML format."""

from __future__ import annotations

import ast
import html
import re
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from ..api.tool_calling import (
    _decode_json_like,
    _get_tool_param_config,
    _schema_type,
    _serialize_tool_arguments,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

_OPEN = "<function"
_CLOSE = "</function>"
_CDATA_OPEN = "<![CDATA["
_CDATA_CLOSE = "]]>"

_NAME = r"""(?:"([^"]*)"|'([^']*)')"""
_FUNCTION_OPEN = re.compile(rf"<function\s+name\s*=\s*{_NAME}\s*(/?)>")
_PARAM_OPEN = re.compile(rf"\s*<param\s+name\s*=\s*{_NAME}\s*(/?)>")
_FUNCTION_CLOSE = re.compile(r"\s*</function>\Z")
# The XML character references a model writes instead of CDATA; a bare ``&`` is data.
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#[0-9]+|#[xX][0-9a-fA-F]+);")


def _tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _attr(match: re.Match[str]) -> str:
    """The quoted ``name`` attribute of a matched ``<function>``/``<param>`` opener."""
    return (match.group(1) if match.group(1) is not None else match.group(2)).strip()


def _structured_value(raw: str, schema: Any) -> Any:
    """A raw parameter value as the array/object the schema declares, else unchanged.

    The MiniCPM5 chat template renders a non-string argument with Python's ``str``
    (``['a', 'b']``, ``{'k': 1}``) and the model writes it the same way, which is
    not JSON. JSON is tried first, then a Python literal (``ast.literal_eval``: literals
    only, nothing is executed); only a list or a dict is accepted, anything else stays
    the raw string.
    """
    if _schema_type(schema) not in ("array", "object"):
        return raw
    decoded = _decode_json_like(raw)
    if not isinstance(decoded, str):
        return decoded
    try:
        literal = ast.literal_eval(raw.strip())
    except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return raw
    return literal if isinstance(literal, list | dict) else raw


def _unescape(text: str) -> str:
    return _ENTITY.sub(lambda m: html.unescape(m.group()), text)


def _read_value(block: str, pos: int) -> tuple[str, int] | None:
    """One parameter value starting after its opening tag, and where it ends.

    Text is data up to ``</param>``; a ``<![CDATA[...]]>`` section inside it is exact
    (no decoding) and may hold ``</param>``. Plain text between sections has its
    well-formed character references decoded. ``None`` if the value never ends.
    """
    parts: list[str] = []
    while True:
        close = block.find("</param>", pos)
        cdata = block.find(_CDATA_OPEN, pos)
        if cdata >= 0 and (close < 0 or cdata < close):
            parts.append(_unescape(block[pos:cdata]))
            end = block.find(_CDATA_CLOSE, cdata + len(_CDATA_OPEN))
            if end < 0:
                return None
            parts.append(block[cdata + len(_CDATA_OPEN) : end])
            pos = end + len(_CDATA_CLOSE)
            continue
        if close < 0:
            return None
        parts.append(_unescape(block[pos:close]))
        return "".join(parts), close + len("</param>")


def _scan_function(block: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Name and raw ``(key, value)`` parameters of one complete ``<function>`` block.

    The chat template asks for ``<![CDATA[...]]>`` around a value holding ``<``,
    ``&`` or a newline, but the model does not always write it, so the block is
    scanned, not parsed as XML: a value is the text up to the next ``</param>``
    (``_read_value``), so ``<``, ``&`` and tags in it are data. ``None`` for a block
    that is not a valid call: no name, an attribute other than ``name``, a duplicate
    or unnamed parameter, text between parameters, a value that never ends.
    """
    opener = _FUNCTION_OPEN.match(block)
    if opener is None:
        return None
    name = _attr(opener)
    if not name:
        return None
    pos = opener.end()
    if opener.group(3):  # <function name="..."/>
        return (name, []) if pos == len(block) else None
    params: list[tuple[str, str]] = []
    seen: set[str] = set()
    while (param := _PARAM_OPEN.match(block, pos)) is not None:
        key = _attr(param)
        if not key or key in seen:
            return None
        seen.add(key)
        pos = param.end()
        if param.group(3):  # <param name="..."/>
            params.append((key, ""))
            continue
        read = _read_value(block, pos)
        if read is None:
            return None
        value, pos = read
        params.append((key, value))
    return (name, params) if _FUNCTION_CLOSE.match(block, pos) else None



@ToolParserManager.register_module("minicpm")
class MiniCPMToolParser(ToolParser):
    """Parse ``<function name=...><param name=...>...</param></function>``."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("minicpm_native",)

    @staticmethod
    def _is_function_opener(text: str, start: int) -> bool:
        """Avoid treating ordinary words such as ``<functionality>`` as XML."""
        end = start + len(_OPEN)
        return end == len(text) or text[end].isspace() or text[end] in ">/"

    @staticmethod
    def _function_end(text: str, start: int) -> int | None:
        """Find the native close tag without mistaking CDATA data for markup."""
        cursor = start + len(_OPEN)
        opener_end = text.find(">", cursor)
        if opener_end != -1 and text[opener_end - 1] == "/":
            return opener_end + 1
        while True:
            close = text.find(_CLOSE, cursor)
            if close == -1:
                return None
            cdata = text.find(_CDATA_OPEN, cursor)
            if cdata == -1 or close < cdata:
                return close + len(_CLOSE)
            cdata_end = text.find(_CDATA_CLOSE, cdata + len(_CDATA_OPEN))
            if cdata_end == -1:
                return None
            cursor = cdata_end + len(_CDATA_CLOSE)

    @classmethod
    def _next_opener_outside_cdata(cls, text: str, pos: int) -> int:
        """Index of the next ``<function`` at/after ``pos`` outside CDATA, or -1.

        Resynchronization after a malformed opener must never land on a
        ``<function`` that lives inside a CDATA *value* — that text is data,
        not markup, and parsing it would fabricate an executable tool call
        from a string literal. Skips over terminated CDATA sections; an
        unterminated CDATA swallows the rest of the text, so nothing valid
        can follow and we report -1.
        """
        i = pos
        while True:
            opener = text.find(_OPEN, i)
            if opener == -1:
                return -1
            cdata = text.find(_CDATA_OPEN, i)
            if cdata != -1 and cdata < opener:
                cdata_end = text.find(_CDATA_CLOSE, cdata + len(_CDATA_OPEN))
                if cdata_end == -1:
                    return -1
                i = cdata_end + len(_CDATA_CLOSE)
                continue
            return opener

    @classmethod
    def _blocks(
        cls, text: str
    ) -> Iterator[tuple[int, int, tuple[str, list[tuple[str, str]]]]]:
        """Yield complete, valid native calls in wire order: ``(start, end, scan)``."""
        cursor = 0
        while (start := text.find(_OPEN, cursor)) != -1:
            if not cls._is_function_opener(text, start):
                cursor = start + len(_OPEN)
                continue

            end = cls._function_end(text, start)
            if end is None:
                # No complete close for this opener: either no ``</function>``
                # at all, or an unterminated CDATA that swallows the rest of
                # the text. In both cases no complete call can follow, and
                # scanning forward would risk parsing CDATA data as markup —
                # so stop.
                return
            scan = _scan_function(text[start:end])
            if scan is None:
                # Not a valid call. ``_function_end`` may have bound a *later*
                # call's close tag to this malformed opener (e.g. a nested
                # ``<function>``). Resync at the next opener that is NOT
                # inside a CDATA value so the nested/later valid call is
                # recovered without ever treating CDATA data as a tool call.
                nxt = cls._next_opener_outside_cdata(text, start + len(_OPEN))
                if nxt == -1:
                    return
                cursor = nxt
                continue
            yield start, end, scan
            cursor = end

    @staticmethod
    def _call_from_scan(
        scan: tuple[str, list[tuple[str, str]]], request: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Type one scanned call's arguments by the tool schema, as OpenAI JSON."""
        name, params = scan
        schema = _get_tool_param_config(name, request)
        arguments = {
            key: _structured_value(raw, schema.get(key)) for key, raw in params
        }
        return {
            "id": _tool_id(),
            "name": name,
            "arguments": _serialize_tool_arguments(arguments, name, request),
        }

    @classmethod
    def _parsed(
        cls, text: str, request: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], str]:
        calls: list[dict[str, Any]] = []
        spans: list[tuple[int, int]] = []
        for start, end, scan in cls._blocks(text):
            calls.append(cls._call_from_scan(scan, request))
            spans.append((start, end))

        if not spans:
            return calls, text
        content: list[str] = []
        cursor = 0
        for start, end in spans:
            content.append(text[cursor:start])
            cursor = end
        content.append(text[cursor:])
        return calls, "".join(content)

    @classmethod
    def _incomplete_start(cls, text: str) -> int | None:
        """Index of the outermost pending native opener, or ``None``.

        Everything from this index onward is in-flight tool markup that
        must stay out of streamed content until the call completes (or is
        proven invalid). Scans forward and is CDATA-aware via
        ``_function_end``: a ``<function`` substring inside an unterminated
        CDATA value belongs to the enclosing call, not a new opener, so the
        outer markup is never mistaken for the pending opener and leaked as
        SSE content mid-call. Returns the FIRST opener with no complete
        close (``rfind`` would pick an inner CDATA occurrence instead).
        """
        cursor = 0
        while (start := text.find(_OPEN, cursor)) != -1:
            if not cls._is_function_opener(text, start):
                cursor = start + len(_OPEN)
                continue
            end = cls._function_end(text, start)
            if end is None:
                return start
            cursor = end
        # No full opener pending: a trailing partial prefix ("<", "<f",
        # "<fun", …) could still grow into an opener, so hold it too.
        for length in range(min(len(_OPEN) - 1, len(text)), 0, -1):
            if text.endswith(_OPEN[:length]):
                return len(text) - length
        return None

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Return the part of an SSE prefix known not to be native markup."""
        start = cls._incomplete_start(text)
        return text if start is None else text[:start]

    def has_pending_tool_call(self, text: str) -> bool:
        return self._incomplete_start(text) is not None

    def flush_held_content(self, full_text: str) -> str:
        # Release exactly what streaming withheld: compute the held tail
        # from the call-stripped residual (``_parsed(...)[1]``, the same
        # representation ``_safe_content_prefix`` runs on during
        # streaming) rather than the raw text. A raw-text scan could drop
        # a malformed residual that streaming held, or re-release bytes
        # that belong to an already-completed call.
        _, residual = self._parsed(full_text, None)
        start = self._incomplete_start(residual)
        return residual[start:] if start is not None else ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        calls, content = self._parsed(model_output, request)
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=(content or None) if calls else model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        # De-duplicate emitted calls statelessly: derive the count of
        # already-streamed calls from ``previous_text`` on every delta
        # rather than caching a running counter. This mirrors
        # ``HermesToolParser`` (which recomputes ``prev_completed`` from
        # ``previous_text``) and is what makes a non-monotonic prefix
        # revision safe — with no counter to reset, a call that
        # ``previous_text`` already contains can never be re-emitted, so
        # an SSE client cannot execute the same tool twice. The
        # postprocessor drives this method with strictly append-only
        # accumulated text (``current = previous + delta``), so the two
        # ``_parsed`` scans stay within the established ABC streaming
        # contract; tool-call bodies are bounded, so the re-scan is
        # dominated by decode cost, same as every sibling parser.
        previous_calls, previous_content = self._parsed(previous_text, request)
        calls, current_content = self._parsed(current_text, request)
        safe_current = self._safe_content_prefix(current_content)
        safe_previous = self._safe_content_prefix(previous_content)
        already_emitted = len(previous_calls)
        new_calls = calls[already_emitted:]
        if new_calls:
            event: dict[str, Any] = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for index, call in enumerate(new_calls, already_emitted)
                ]
            }
            if safe_current.startswith(safe_previous):
                content = safe_current[len(safe_previous) :]
            else:
                content = safe_current
            if content:
                event["content"] = content
            return event

        if safe_current.startswith(safe_previous):
            delta = safe_current[len(safe_previous) :]
        else:
            delta = safe_current
        return {"content": delta} if delta else None
