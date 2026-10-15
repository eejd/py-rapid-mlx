#!/usr/bin/env python3
"""Start isolated app-equivalent Goose proxy/bridge/helper for the QA suite."""
import argparse
import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = pathlib.Path(__file__).parent


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ready(port, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Helper exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=.5):
                return
        except Exception:
            time.sleep(.15)
    raise TimeoutError(f"Helper on port {port} did not become ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=pathlib.Path, required=True, help="staged app Contents/Resources, never installed app")
    parser.add_argument("--engine-port", type=int, required=True, help="own 27B MTP server port")
    parser.add_argument("--engine-bearer-env", default="GOOSE_QA_ENGINE_BEARER")
    parser.add_argument("--model", default="qwen3.6-27b-4bit")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--budget-min", type=float, default=45)
    args = parser.parse_args()
    engine_bearer = os.environ[args.engine_bearer_env]
    resources = args.resources.resolve()
    goose = resources / "Helpers/goose"
    proxy = resources / "Helpers/work-inference-proxy.py"
    watchdog = resources / "Helpers/work-helper-watchdog.py"
    web = resources / "Helpers/chat-web-mcp.py"
    python = resources / "rapid-mlx/python/bin/python3"
    for path in (goose, proxy, watchdog, web, python):
        if not path.is_file():
            parser.error(f"Missing staged helper: {path}")
    own = pathlib.Path(tempfile.mkdtemp(prefix="goose-qa-"))
    own.chmod(0o700)
    (own / "tmp").mkdir()
    provider_dir = own / "config/custom_providers"
    provider_dir.mkdir(parents=True)
    proxy_port, bridge_port, goose_port = free_port(), free_port(), free_port()
    secret, scoped_key = secrets.token_hex(32), secrets.token_hex(32)
    provider = {"name": "custom_rapid_local", "engine": "openai", "display_name": "Rapid Local",
                "api_key_env": "RAPID_LOCAL_WORK_KEY", "base_url": f"http://127.0.0.1:{bridge_port}/v1/chat/completions",
                "models": [{"name": args.model, "context_limit": 32768}], "supports_streaming": True,
                "requires_auth": True, "dynamic_models": False}
    (provider_dir / "custom_rapid_local.json").write_text(json.dumps(provider))
    base_env = {"HOME": str(own), "USER": os.environ.get("USER", "eval"),
                "LOGNAME": os.environ.get("LOGNAME", "eval"), "LANG": "en_US.UTF-8",
                "NO_PROXY": "127.0.0.1,localhost", "GOOSE_PATH_ROOT": str(own),
                "GOOSE_SHELL": str(watchdog), "GOOSE_TELEMETRY_OFF": "1",
                "GOOSE_TELEMETRY_ENABLED": "false", "GOOSE_DISABLE_KEYRING": "1",
                "GOOSE_PROVIDER": "custom_rapid_local", "GOOSE_MODEL": args.model,
                "GOOSE_MODE": "approve", "RAPID_LOCAL_WORK_KEY": scoped_key,
                "GOOSE_SERVER__SECRET_KEY": secret, "WORK_HELPER_OWNERSHIP_TOKEN": secrets.token_hex(32),
                "PATH": str(python.parent) + ":/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONPATH": str(resources / "rapid-mlx/site-packages"),
                "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(own / "tmp")}
    processes = []
    def launch(command, env, name):
        log = (own / (name + ".log")).open("wb")
        process = subprocess.Popen(command, cwd=own, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log))
        return process
    try:
        proxy_env = dict(base_env)
        proxy_env.pop("GOOSE_SERVER__SECRET_KEY")
        proxy_env.pop("RAPID_LOCAL_WORK_KEY")
        proxy_env["WORK_PROXY_BEARER"] = engine_bearer
        proxy_process = launch([str(python), "-I", "-S", str(proxy), str(proxy_port), str(args.engine_port), args.model,
                                "--full-access", "--goose-chat"], proxy_env, "proxy")
        ready(proxy_port, proxy_process)
        bridge_env = dict(base_env)
        bridge_env.pop("GOOSE_SERVER__SECRET_KEY")
        bridge_env.pop("RAPID_LOCAL_WORK_KEY")
        bridge_env["WORK_BRIDGE_SCOPED_KEY"] = scoped_key
        bridge_env["WORK_BRIDGE_UPSTREAM_KEY"] = engine_bearer
        bridge_process = launch([str(python), "-I", "-S", str(watchdog), "--credential-bridge",
                                 str(bridge_port), str(proxy_port)], bridge_env, "bridge")
        ready(bridge_port, bridge_process)
        goose_process = launch([str(goose), "serve", "--host", "127.0.0.1", "--port", str(goose_port),
                                "--platform", "cli", "--with-builtin",
                                "developer,computercontroller,memory,autovisualiser,tutorial"], base_env, "goose")
        ready(goose_port, goose_process)
        runner_env = dict(os.environ)
        runner_env["GOOSE_QA_ACP_SECRET"] = secret
        runner_env["GOOSE_QA_ENGINE_BEARER"] = engine_bearer
        command = [sys.executable, str(HERE / "run.py"), "--acp", f"http://127.0.0.1:{goose_port}",
                   "--engine", f"http://127.0.0.1:{args.engine_port}", "--model", args.model,
                   "--web-script", str(web), "--output", str(args.output.resolve()),
                   "--workspaces", str(own / "workspaces"), "--budget-min", str(args.budget_min)]
        print(f"Isolated eval work: {own}; Goose PID {goose_process.pid}; proxy PID {proxy_process.pid}", flush=True)
        return subprocess.call(command, env=runner_env)
    finally:
        for process, _ in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process, log in reversed(processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()

if __name__ == "__main__":
    raise SystemExit(main())
