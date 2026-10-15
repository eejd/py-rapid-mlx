"""Real-process checks for escaped descendants of the Goose helper."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WATCHDOG = Path(__file__).resolve().parents[2] / "Work/work-helper-watchdog.py"
HELPER = """import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
with open(sys.argv[1], "w") as output: output.write(str(child.pid))
while True: time.sleep(0.1)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _ready(port):
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
    except OSError:
        return False


class WatchdogTest(unittest.TestCase):
    def test_fast_detach_is_found_by_ownership_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "pid"
            token = "watchdog-test-" + str(os.getpid())
            env = dict(os.environ, WORK_HELPER_OWNERSHIP_TOKEN=token)
            helper = subprocess.Popen(
                [
                    sys.executable,
                    str(WATCHDOG),
                    str(os.getpid()),
                    sys.executable,
                    "-c",
                    """import os, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
with open(sys.argv[1], "w") as output: output.write(str(child.pid))
""",
                    str(record),
                ],
                env=env,
            )
            self.assertTrue(wait_for(record.exists))
            child = int(record.read_text())
            try:
                helper.wait(timeout=3)
                os.killpg(helper.pid, signal.SIGTERM)
                self.assertTrue(wait_for(lambda: not alive(child)))
            finally:
                if alive(child):
                    os.kill(child, signal.SIGKILL)

    def test_bridge_scopes_inference_credential(self):
        seen = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                seen.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        import threading

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        bridge_port = 0
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            bridge_port = sock.getsockname()[1]
        env = dict(
            os.environ,
            WORK_BRIDGE_SCOPED_KEY="scoped-test-key",
            WORK_BRIDGE_UPSTREAM_KEY="real-engine-bearer",
        )
        contract_dir = tempfile.TemporaryDirectory()
        (Path(contract_dir.name) / "chat-contract.json").write_text(
            '{"system_prompt":"","use_goose_sampling_defaults":true}'
        )
        bridge = subprocess.Popen(
            [
                sys.executable,
                str(WATCHDOG),
                "--credential-bridge",
                str(bridge_port),
                str(server.server_port),
            ],
            env=env,
            cwd=contract_dir.name,
        )
        url = f"http://127.0.0.1:{bridge_port}/v1/chat/completions"
        try:
            self.assertTrue(wait_for(lambda: _ready(bridge_port)))
            with self.assertRaises(HTTPError) as rejected:
                urlopen(
                    Request(url, b"{}", {"Authorization": "Bearer real-engine-bearer"})
                )
            self.assertEqual(rejected.exception.code, 403)
            rejected.exception.close()
            response = urlopen(
                Request(url, b"{}", {"Authorization": "Bearer scoped-test-key"})
            )
            self.assertEqual(response.read(), b'{"ok":true}')
            self.assertEqual(seen, ["Bearer real-engine-bearer"])
        finally:
            bridge.terminate()
            bridge.wait()
            server.shutdown()
            server.server_close()
            contract_dir.cleanup()

    def test_bridge_forwards_first_stream_chunk_immediately(self):
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.wfile.flush()
                time.sleep(0.5)
                self.wfile.write(b"data: last\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        import threading

        threading.Thread(target=server.serve_forever, daemon=True).start()
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(
            os.environ,
            WORK_BRIDGE_SCOPED_KEY="scoped-test-key",
            WORK_BRIDGE_UPSTREAM_KEY="real-engine-bearer",
        )
        contract_dir = tempfile.TemporaryDirectory()
        (Path(contract_dir.name) / "chat-contract.json").write_text(
            '{"system_prompt":"","use_goose_sampling_defaults":true}'
        )
        bridge = subprocess.Popen(
            [
                sys.executable,
                str(WATCHDOG),
                "--credential-bridge",
                str(port),
                str(server.server_port),
            ],
            env=env,
            cwd=contract_dir.name,
        )
        try:
            self.assertTrue(wait_for(lambda: _ready(port)))
            start = time.monotonic()
            with urlopen(
                Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    b"{}",
                    {"Authorization": "Bearer scoped-test-key"},
                )
            ) as response:
                self.assertEqual(response.readline(), b"data: first\n")
            self.assertLess(time.monotonic() - start, 0.4)
        finally:
            bridge.terminate()
            bridge.wait()
            server.shutdown()
            server.server_close()
            contract_dir.cleanup()

    def test_shell_drops_helper_credentials(self):
        env = dict(
            os.environ,
            GOOSE_SERVER__SECRET_KEY="acp-secret",
            RAPID_LOCAL_WORK_KEY="inference-secret",
            WORK_PROXY_BEARER="engine-secret",
        )
        result = subprocess.run(
            [str(WATCHDOG), "-c", "env"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        for secret in ("acp-secret", "inference-secret", "engine-secret"):
            self.assertNotIn(secret, result.stdout)

    def test_stop_reaps_setsid_child(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "pid"
            helper = subprocess.Popen(
                [
                    sys.executable,
                    str(WATCHDOG),
                    str(os.getpid()),
                    sys.executable,
                    "-c",
                    HELPER,
                    str(record),
                ]
            )
            self.assertTrue(wait_for(record.exists))
            child = int(record.read_text())
            try:
                time.sleep(0.3)
                os.killpg(helper.pid, signal.SIGTERM)
                self.assertTrue(wait_for(lambda: not alive(child)))
            finally:
                if alive(child):
                    os.kill(child, signal.SIGKILL)
                if helper.poll() is None:
                    os.killpg(helper.pid, signal.SIGKILL)
                helper.wait()

    def test_owner_sigkill_reaps_setsid_child(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "pid"
            owner = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    """import os, subprocess, sys, time
subprocess.Popen([sys.executable, sys.argv[1], str(os.getpid()), sys.executable,
                  "-c", sys.argv[2], sys.argv[3]])
while True: time.sleep(0.1)
""",
                    str(WATCHDOG),
                    HELPER,
                    str(record),
                ]
            )
            self.assertTrue(wait_for(record.exists))
            child = int(record.read_text())
            try:
                time.sleep(0.3)
                owner.kill()
                owner.wait()
                self.assertTrue(wait_for(lambda: not alive(child)))
            finally:
                if alive(child):
                    os.kill(child, signal.SIGKILL)
                if owner.poll() is None:
                    owner.kill()


if __name__ == "__main__":
    unittest.main()
