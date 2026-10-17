# SPDX-License-Identifier: Apache-2.0
"""CR, LF and split CRLF delimit Markdown examples consistently (#4788)."""

from unittest.mock import patch

import pytest

from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
    _MarkdownCodeTracker,
)
from tests.qwen3coder_stream_harness import deltas, non_stream, stream
from tests.test_qwen3coder_fenced_declared_4714 import BARE, CALL, REQUEST

BREAKS = pytest.mark.parametrize("newline", ["\r", "\r\n", "\n"])
SIZES = pytest.mark.parametrize("size", [None, 1, 2, 3, 7, 0])


@BREAKS
@SIZES
@pytest.mark.parametrize("marker", ["```", "~~~"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("real_call", [False, True])
def test_fenced_example_parity(newline, size, marker, wrapped, real_call):
    call = (CALL if wrapped else BARE).replace("\n", newline)
    example = newline.join(["Example:", marker + "xml", call, marker, ""])
    text = example + (CALL if real_call else "")
    expected = ([("get_weather", '{"city": "Paris"}')] if real_call else [], example)
    result = non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    assert result == expected


def test_harness_splits_crlf():
    pieces = deltas("```xml\r\n" + CALL, 1)
    index = pieces.index("\r")
    assert pieces[index + 1] == "\n"


@BREAKS
@SIZES
def test_truncated_fenced_example_stays_content(newline, size):
    text = newline.join(["Example:", "```xml", "<tool_call>", "<function=get_weather>"])
    result = non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    assert result == ([], text)


@BREAKS
def test_buffered_cleanup_after_closed_fence(newline):
    prefix = newline.join(["```xml", "example", "```", ""])
    text = prefix + "<tool_call>"
    assert Qwen3CoderToolParser.strip_trailing_tool_envelope(text) == prefix
    fenced = newline.join(["```xml", "<tool_call>"])
    assert Qwen3CoderToolParser.strip_trailing_tool_envelope(fenced) == fenced


@BREAKS
@pytest.mark.parametrize("size", [1, 2, 3, 7, 0])
def test_cross_format_fallback_mask(newline, size):
    example = newline.join(
        ["~~~xml", '[Calling tool: get_weather({"city":"Paris"})]', "~~~", ""]
    )
    tail = '[Calling tool: get_weather({"city":"Rome"})]'
    masked = Qwen3CoderToolParser.mask_markdown_fenced_regions(example + tail)
    assert masked == "".join(c if c in "\r\n" else " " for c in example) + tail
    assert stream(example, REQUEST, size)[0] == []
    calls, _ = stream(example + tail, REQUEST, size)
    assert calls == [("get_weather", '{"city": "Rome"}')]


@BREAKS
@pytest.mark.parametrize("method", ["_feed", "_feed_fence"])
def test_inline_and_fence_line_state(newline, method):
    tracker = _MarkdownCodeTracker()
    feed = getattr(tracker, method)
    for char in "`inline" + newline + "```xml" + newline:
        feed(char)
    result = tracker._result if method == "_feed" else tracker._fence_result
    assert result()
    for char in "body" + newline + "```" + newline:
        feed(char)
    assert not result()
    if method == "_feed":
        feed("`inline" + newline)
        assert not result()


@pytest.mark.parametrize("plain", [False, True])
def test_split_crlf_finishes_only_once(plain):
    tracker = _MarkdownCodeTracker()
    with patch.object(
        tracker, "_finish_fence_line", wraps=tracker._finish_fence_line
    ) as end:
        tracker._feed_fence("```xml\r")
        assert end.call_count == 1
        if plain:
            tracker._feed_fence_plain("\n")
        else:
            tracker._feed_fence("\n")
        assert end.call_count == 1
        tracker._feed_fence("x\r\n```\r\n")
        assert end.call_count == 3
    assert not tracker._fence_result()


@BREAKS
def test_skipped_call_newline_allows_next_fence(newline):
    tracker = _MarkdownCodeTracker()
    text = BARE + newline + "```xml" + newline
    tracker.skip_fence_span(text, 0, len(BARE + newline))
    assert tracker.inside_fence(text, len(text))


@pytest.mark.parametrize("method", ["inside", "inside_fence"])
def test_speculative_cr_state_is_restored(method):
    tracker = _MarkdownCodeTracker()
    text = "<tool_call>\r\n```xml\r\n"
    query = getattr(tracker, method)
    assert not query(text, len("<tool_call>\r"))
    assert query(text, len(text))
    # The earlier query must also work from scratch.
    assert not query(text, len("<tool_call>\r"))
