#!/usr/bin/env python3
"""Run fixed QA items through a running Goose ACP helper and direct engine path.

The caller starts Goose with the same provider/proxy/extensions as WorkAgent,
in an isolated HOME, on private ports. This runner never launches or touches an
installed app or shared model server. See README for setup.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import queue
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".cache/rapid-mlx/chat-eval"
BEHAVIOR = ("Reply in the user's language. When the requested task is done, stop and summarize; "
            "do not re-run or re-verify completed work unless asked. Prefer paths relative to "
            "the selected chat folder in replies.\n")


def post(url, body, headers, timeout=30, parse=True):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return (json.load(response) if parse else None), response.headers


def stream(url, body, headers, on_event, timeout=180):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if line.startswith(b"data:"):
                try:
                    event = json.loads(line[5:])
                    if on_event(event) is False:
                        break
                except json.JSONDecodeError:
                    continue


def load_items():
    manifest = json.loads((HERE / "subset.json").read_text())
    for name, source in manifest["sources"].items():
        cache = CACHE / (name + (".parquet" if name == "gdpval" else ".csv" if name == "simpleqa" else ".jsonl"))
        if name == "gaia":
            cache = CACHE / "gaia-metadata.level1.parquet"
        if not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"Missing or changed source data: {name}; run prepare.py")
    ids = manifest["ids"]
    rows = {}
    for name in ("arena", "chinese", "ifeval"):
        rows[name] = {}
        for line in (CACHE / (name + ".jsonl")).read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[name][str(row[{"arena": "uid", "chinese": "id", "ifeval": "key"}[name]])] = row
    rows["simpleqa"] = {}
    for row in csv.DictReader(io.StringIO((CACHE / "simpleqa.csv").read_text())):
        key = hashlib.sha256((row["problem"] + "\0" + row["answer"]).encode()).hexdigest()[:20]
        rows["simpleqa"][key] = row
    try:
        import pyarrow.parquet as pq
        rows["gdpval"] = {row["task_id"]: row for row in pq.read_table(CACHE / "gdpval.parquet").to_pylist()}
        gaia_path = CACHE / "gaia-metadata.level1.parquet"
        if gaia_path.exists():
            rows["gaia"] = {row["task_id"]: row for row in pq.read_table(gaia_path).to_pylist()}
    except ImportError:
        rows["gdpval"] = {}
    for name, keys in ids.items():
        for key in keys:
            row = rows.get(name, {}).get(str(key))
            if row is None:
                continue
            field = {"arena": "prompt", "simpleqa": "problem", "chinese": "question",
                     "ifeval": "prompt", "gaia": "Question", "gdpval": "prompt"}[name]
            yield name, str(key), row[field], row


class ACP:
    def __init__(self, base, secret):
        self.url = base.rstrip("/") + "/acp"
        self.secret = secret
        self.connection = ""
        self.events = queue.Queue()
        self.serial = 0
        self.streams = []
        response, headers = post(self.url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": 1, "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
                       "clientInfo": {"name": "Rapid Work", "version": "0.1"}}}, {"X-Secret-Key": secret})
        if response.get("error"):
            raise RuntimeError(response["error"])
        self.connection = headers["Acp-Connection-Id"]
        self.open_events(None)

    def headers(self, session=None):
        result = {"X-Secret-Key": self.secret, "Acp-Connection-Id": self.connection}
        if session:
            result["Acp-Session-Id"] = session
        return result

    def open_events(self, session):
        ready = threading.Event()
        def read():
            request = urllib.request.Request(self.url, headers={"Accept": "text/event-stream", **self.headers(session)})
            try:
                with urllib.request.urlopen(request, timeout=1200) as response:
                    ready.set()
                    for line in response:
                        if line.startswith(b"data:"):
                            try:
                                self.events.put(json.loads(line[5:]))
                            except json.JSONDecodeError:
                                pass
            except Exception as error:
                self.events.put({"stream_error": str(error)})
            finally:
                ready.set()
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        self.streams.append(thread)
        if not ready.wait(10):
            raise TimeoutError("ACP event stream did not open")

    def send(self, body, session=None):
        post(self.url, body, self.headers(session), parse=False)

    def call(self, method, params, session=None, timeout=180, on_event=None):
        self.serial += 1
        id_ = self.serial + 1
        self.send({"jsonrpc": "2.0", "id": id_, "method": method, "params": params}, session)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=max(.1, min(2, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if event.get("id") == id_ and "method" not in event:
                if event.get("error"):
                    raise RuntimeError(event["error"])
                return event.get("result", {})
            if event.get("method") == "session/request_permission":
                params = event.get("params", {})
                choices = params.get("options", [])
                option = next((x.get("optionId") for x in choices if x.get("kind") == "allow_once"), None)
                if option:
                    self.send({"jsonrpc": "2.0", "id": event["id"],
                               "result": {"outcome": {"outcome": "selected", "optionId": option}}}, params.get("sessionId"))
            if on_event:
                on_event(event)
        raise TimeoutError(f"{method} timed out")

    def ask(self, question, folder, web_script, timeout):
        servers = [{"name": "web", "command": "/usr/bin/python3", "args": ["-I", "-S", str(web_script)], "env": []}] if web_script else []
        session = self.call("session/new", {"cwd": str(folder), "mcpServers": servers,
                                             "_meta": {"client": "Rapid Work"}})["sessionId"]
        self.open_events(session)
        state = {"answer": "", "first_visible_s": None, "tool_calls": 0, "permission_prompts": 0,
                 "failure": None, "empty": False}
        seen_tools = set()
        start = time.monotonic()
        def update(event):
            if event.get("method") == "session/request_permission":
                state["permission_prompts"] += 1
            if event.get("method") != "session/update":
                return
            change = event.get("params", {}).get("update", {})
            kind = change.get("sessionUpdate")
            if kind == "agent_message_chunk":
                chunk = change.get("content", {}).get("text", "")
                if chunk and state["first_visible_s"] is None:
                    state["first_visible_s"] = round(time.monotonic() - start, 3)
                state["answer"] += chunk
            if kind == "tool_call" and change.get("toolCallId") not in seen_tools:
                seen_tools.add(change.get("toolCallId"))
                state["tool_calls"] += 1
        try:
            self.call("session/prompt", {"sessionId": session, "prompt": [{"type": "text", "text": BEHAVIOR + question}]},
                      session=session, timeout=timeout, on_event=update)
        except Exception as error:
            state["failure"] = str(error)
        state["total_s"] = round(time.monotonic() - start, 3)
        state["empty"] = not state["answer"].strip()
        state["generated_files"] = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()]
        return state


def direct(question, base, model, bearer, timeout):
    state = {"answer": "", "first_visible_s": None, "tool_calls": 0, "permission_prompts": 0,
             "failure": None, "empty": False}
    start = time.monotonic()
    def update(event):
        for choice in event.get("choices", []):
            chunk = choice.get("delta", {}).get("content") or ""
            if chunk and state["first_visible_s"] is None:
                state["first_visible_s"] = round(time.monotonic() - start, 3)
            state["answer"] += chunk
    try:
        stream(base.rstrip("/") + "/v1/chat/completions", {"model": model,
            "messages": [{"role": "user", "content": question}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 1024},
            {"Authorization": "Bearer " + bearer}, update, timeout)
    except Exception as error:
        state["failure"] = str(error)
    state["total_s"] = round(time.monotonic() - start, 3)
    state["empty"] = not state["answer"].strip()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acp", required=True, help="isolated Goose helper base URL")
    parser.add_argument("--secret-env", default="GOOSE_QA_ACP_SECRET", help="name of env var with helper secret")
    parser.add_argument("--engine", required=True, help="own 27B engine base URL")
    parser.add_argument("--bearer-env", default="GOOSE_QA_ENGINE_BEARER", help="name of env var with engine bearer")
    parser.add_argument("--model", default="qwen3.6-27b-4bit")
    parser.add_argument("--web-script", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workspaces", type=pathlib.Path, required=True)
    parser.add_argument("--budget-min", type=float, default=45)
    parser.add_argument("--item-timeout", type=int, default=120)
    args = parser.parse_args()
    if not args.web_script or not args.web_script.is_file():
        parser.error("--web-script must point to the same bundled MCP script used by the app")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.workspaces.mkdir(parents=True, exist_ok=True)
    secret = os.environ[args.secret_env]
    bearer = os.environ[args.bearer_env]
    acp = ACP(args.acp, secret)
    deadline = time.monotonic() + args.budget_min * 60
    completed = {(r["benchmark"], r["id"], r["path"], bool(r["web"])) for line in args.output.read_text().splitlines()
                 if line.strip() for r in [json.loads(line)]} if args.output.exists() else set()
    with args.output.open("a") as output:
        for benchmark, id_, question, _ in load_items():
            paths = [("goose", True), ("goose", False), ("direct", False)] if benchmark in ("simpleqa", "chinese") else [("goose", True), ("direct", False)]
            for path, web in paths:
                if (benchmark, id_, path, web) in completed:
                    continue
                if time.monotonic() >= deadline:
                    return
                folder = args.workspaces / (benchmark + "-" + id_)
                folder.mkdir(parents=True, exist_ok=True)
                result = (acp.ask(question, folder, args.web_script if web else None, args.item_timeout)
                          if path == "goose" else direct(question, args.engine, args.model, bearer, args.item_timeout))
                result.update({"benchmark": benchmark, "id": id_, "path": path, "web": web})
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(benchmark, id_, path, web, result["total_s"], result["failure"], flush=True)

if __name__ == "__main__":
    main()
