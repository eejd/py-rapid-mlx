"""Bounded stdin/stdout harness for a real native endpoint, never a network API."""

from __future__ import annotations

import json
import pathlib
import select
import subprocess


class NativeProvider:
    def __init__(self, executable: pathlib.Path) -> None:
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        try:
            self.hello = self._read(timeout=20)
            if self.hello.get("type") != "hello":
                raise RuntimeError("native_start_failed")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: float = 5) -> dict:
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise RuntimeError("native_timeout")
        line = self.process.stdout.readline(65537)
        if not line.endswith("\n") or len(line) > 65536:
            raise RuntimeError("invalid_native_frame")
        return json.loads(line)

    def call(self, request: dict) -> dict:
        line = json.dumps(request, separators=(",", ":"), ensure_ascii=False)
        if len(line.encode()) > 65536:
            raise ValueError("frame_too_large")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
