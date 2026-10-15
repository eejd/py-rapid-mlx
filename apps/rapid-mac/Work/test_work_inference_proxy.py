"""Focused regression test for the Goose Chat inference rate while streaming."""

import importlib.util
import io
import json
import os
import pathlib
import tempfile
import sys
import unittest
from unittest.mock import patch


PROXY = pathlib.Path(__file__).with_name("work-inference-proxy.py")


class StreamRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("work_inference_proxy", PROXY)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "argv", [str(PROXY), "0", "0", "test-model"]), patch.dict(
            os.environ, {"WORK_PROXY_BEARER": "test"}
        ):
            spec.loader.exec_module(cls.module)

    def test_rate_is_available_before_terminal_usage(self):
        module = self.module
        rate = module.StreamRate()
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]):
            rate.feed(b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n')
            self.assertIsNone(rate.live_tokens_per_second())
            rate.feed(b'data: {"choices":[{"delta":{"content":"world!"}}]}\n\n')

        self.assertEqual(rate.live_tokens_per_second(), 4.0)
        self.assertIsNone(rate.tokens_per_second())
        rate.feed(b'data: {"choices":[],"usage":{"completion_tokens":11}}\n\n')
        self.assertEqual(rate.tokens_per_second(), 20.0)

    def test_goose_request_removes_global_prompt_and_unused_tool_schemas(self):
        original = {
            "model": "test-model", "stream": True,
            "messages": [
                {"role": "system", "content": "global skills " * 2000},
                {"role": "user", "content": "今天谷歌股价多少？"},
            ],
            "tools": [{"type": "function", "function": {"name": name,
                       "description": "large schema " * 50}}
                      for name in ("shell", "tree", "write", "edit", "apps__create_app",
                                   "autovisualiser__show_chart")],
        }
        compact = self.module.compact_goose_request(json.loads(json.dumps(original)))
        self.assertLess(len(json.dumps(compact)), 4000)
        self.assertEqual(compact["messages"][1], original["messages"][1])
        self.assertEqual([t["function"]["name"] for t in compact["tools"]],
                         ["shell", "tree", "write", "edit"])
        self.assertIn("current facts", compact["messages"][0]["content"])
        self.assertIn("exact source URL", compact["messages"][0]["content"])
        self.assertIn("GOOGL is Alphabet Class A", compact["messages"][0]["content"])
        self.assertIn("google.com/finance/quote/GOOGL:NASDAQ", compact["messages"][0]["content"])

    def test_rotated_server_key_is_read_for_each_upstream_request(self):
        module = self.module
        with tempfile.TemporaryDirectory() as directory:
            key_file = pathlib.Path(directory) / "key"
            with patch.object(module, "BEARER_FILE", str(key_file)):
                key_file.write_text("first")
                self.assertEqual(module.upstream_bearer(), "first")
                key_file.write_text("rotated")
                self.assertEqual(module.upstream_bearer(), "rotated")

    def test_tool_arguments_contribute_to_live_rate(self):
        module = self.module
        rate = module.StreamRate()
        with patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]):
            rate.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"write_file","arguments":"{\\\"path\\\":\\\"a"}}]}}]}\n\n')
            rate.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":".txt\\\"}"}}]}}]}\n\n')
        self.assertGreater(rate.live_tokens_per_second(), 0)

    def test_older_request_cannot_overwrite_newer_rate(self):
        module = self.module
        previous = module.LATEST_METRIC.copy()
        try:
            module.LATEST_METRIC.update(sequence=2, tokensPerSecond=5.0)
            module.publish_metric(1, 99.0)
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 5.0)
            module.publish_metric(2, 20.0)
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
        finally:
            module.LATEST_METRIC.update(previous)

    def test_terminal_usage_is_published_before_upstream_eof(self):
        module = self.module
        previous = module.LATEST_METRIC.copy()
        class Response:
            status = 200
            headers = {"Content-Type": "text/event-stream"}
            index = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read1(self, _size):
                self.index += 1
                if self.index == 1:
                    return b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                if self.index == 2:
                    return b'data: {"choices":[{"delta":{"content":"world!"}}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":11}}\n\ndata: [DONE]\n\n'
                self_test.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
                return b""

        self_test = self
        body = json.dumps({"model": "test-model", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        handler = object.__new__(module.Handler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Authorization": "Bearer test", "Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        try:
            module.LATEST_METRIC.update(sequence=0, tokensPerSecond=None)
            with patch.object(module, "GOOSE_CHAT", True), patch.object(
                module.OPENER, "open", return_value=Response()
            ), patch.object(module.time, "monotonic", side_effect=[10.0, 10.5]):
                handler.do_POST()
            self.assertEqual(module.LATEST_METRIC["tokensPerSecond"], 20.0)
        finally:
            module.LATEST_METRIC.update(previous)


if __name__ == "__main__":
    unittest.main()
