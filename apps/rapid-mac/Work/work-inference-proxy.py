#!/usr/bin/env python3
"""Text-only loopback gate between the Work helper and Rapid inference.

The helper is sandboxed away from Rapid's real port. This gate rejects media
parts before forwarding a chat request, so Rapid cannot fetch a remote URL on
the helper's behalf through its multimodal input handlers.
"""

import hmac
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(sys.argv[1])
RAPID_PORT = int(sys.argv[2])
MODEL = sys.argv[3]
FULL_ACCESS = len(sys.argv) > 4 and sys.argv[4] == "--full-access"
BEARER = os.environ["WORK_PROXY_BEARER"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def text_only(body):
    if not isinstance(body, dict) or body.get("model") != MODEL:
        return False
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if body.get("modalities") not in (None, ["text"]):
        return False
    if any(key in body for key in ("input", "image", "audio", "video", "files")):
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in (
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        ):
            return False
        if any(
            key in message
            for key in ("image_url", "audio_url", "video_url", "file_url")
        ):
            return False
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False
        if any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            or set(part) != {"type", "text"}
            for part in content
        ):
            return False
    return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + BEARER
        ):
            self.send_error(401)
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_error(400)
            return
        if not 0 < length <= (100_000_000 if FULL_ACCESS else 8_000_000):
            self.send_error(413)
            return
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self.send_error(400)
            return
        if not FULL_ACCESS and not text_only(body):
            self.send_error(403, "Work allows text-only inference")
            return
        request = urllib.request.Request(
            f"http://127.0.0.1:{RAPID_PORT}/v1/chat/completions",
            raw,
            {
                "Authorization": "Bearer " + BEARER,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            response = OPENER.open(request, timeout=900)
        except urllib.error.HTTPError as error:
            response = error
        except Exception:
            self.send_error(502, "Local inference unavailable")
            return
        with response:
            self.send_response(response.status)
            self.send_header(
                "Content-Type", response.headers.get("Content-Type", "application/json")
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while chunk := response.read1(16384):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler).serve_forever()
