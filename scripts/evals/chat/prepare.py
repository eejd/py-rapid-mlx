#!/usr/bin/env python3
"""Freeze benchmark IDs; keep source records in a private cache, not Git."""
import csv
import hashlib
import io
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).parent
CACHE = pathlib.Path.home() / ".cache/rapid-mlx/chat-eval"
SOURCES = {
    "arena": "https://raw.githubusercontent.com/lmarena/arena-hard-auto/196f6b826783b3da7310e361a805fa36f0be83f3/data/arena-hard-v2.0/question.jsonl",
    "simpleqa": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
    "chinese": "https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA/resolve/main/chinese_simpleqa.jsonl",
    "ifeval": "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl",
    "gdpval": "https://huggingface.co/datasets/openai/gdpval/resolve/11e7900cdcac61bc4daf59e65feb238acda98fbf/data/train-00000-of-00001.parquet",
}
COUNTS = {"arena": 40, "simpleqa": 30, "chinese": 20, "ifeval": 40, "gdpval": 5}


def ranked(rows, key, count):
    return [str(row[key]) for row in sorted(rows, key=lambda row: hashlib.sha256(("goose-chat-qa-20261015:" + str(row[key])).encode()).hexdigest())[:count]]


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": "goose-chat-qa-20261015", "sources": {}, "ids": {}}
    for name, url in SOURCES.items():
        path = CACHE / (name + (".parquet" if name == "gdpval" else ".csv" if name == "simpleqa" else ".jsonl"))
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        manifest["sources"][name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if name == "gdpval":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
            rows = [r for r in rows if len(r["deliverable_files"]) == 1 and r["deliverable_files"][0].lower().endswith((".docx", ".xlsx")) and not r["reference_files"]]
            # A fixed public gold subset with three documents and two spreadsheets.
            docs = [r for r in rows if r["deliverable_files"][0].lower().endswith(".docx")]
            sheets = [r for r in rows if r["deliverable_files"][0].lower().endswith(".xlsx")]
            manifest["ids"][name] = ranked(docs, "task_id", 3) + ranked(sheets, "task_id", 2)
            continue
        if name == "simpleqa":
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            for row in rows:
                row["id"] = hashlib.sha256((row["problem"] + "\0" + row["answer"]).encode()).hexdigest()[:20]
            key = "id"
        else:
            rows = []
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Source contains a few malformed multiline records.
            key = {"arena": "uid", "chinese": "id", "ifeval": "key"}[name]
        if name == "arena":
            rows = [r for r in rows if r["category"] == "hard_prompt"]
        manifest["ids"][name] = ranked(rows, key, COUNTS[name])
    # GAIA is gated. After the evaluator accepts its access terms, supply a
    # private metadata.level1.parquet path to materialize these 15 fixed IDs.
    gaia = CACHE / "gaia-metadata.level1.parquet"
    if gaia.exists():
        import pyarrow.parquet as pq
        rows = pq.read_table(gaia).to_pylist()
        manifest["ids"]["gaia"] = ranked(rows, "task_id", 15)
        manifest["sources"]["gaia"] = {"url": "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/682dd723ee1e1697e00360edccf2366dc8418dd9/2023/validation/metadata.level1.parquet", "sha256": hashlib.sha256(gaia.read_bytes()).hexdigest()}
    else:
        manifest["ids"]["gaia"] = []
        manifest["gaia_status"] = "Gated validation metadata unavailable; accept source terms and place file in private cache."
    (ROOT / "subset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print({name: len(ids) for name, ids in manifest["ids"].items()})

if __name__ == "__main__":
    prepare()
