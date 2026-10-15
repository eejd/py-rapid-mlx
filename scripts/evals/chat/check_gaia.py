#!/usr/bin/env python3
"""Score gated GAIA validation with the official quasi-exact scorer."""
import argparse
import importlib.util
import json
import pathlib
import re
import urllib.request

REV = "9f133d71362e77b3539f1514f31b9c101a545fec"
CACHE = pathlib.Path.home() / ".cache/rapid-mlx/chat-eval"


def scorer():
    target = CACHE / ("gaia-scorer-" + REV + ".py")
    if not target.exists():
        url = f"https://huggingface.co/spaces/gaia-benchmark/leaderboard/resolve/{REV}/scorer.py"
        target.write_bytes(urllib.request.urlopen(url, timeout=20).read())
    spec = importlib.util.spec_from_file_location("official_gaia_scorer", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.question_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    allowed = set(json.loads((pathlib.Path(__file__).parent / "subset.json").read_text())["ids"]["gaia"])
    gold = {row["task_id"]: row["Final answer"] for row in
            pq.read_table(CACHE / "gaia-metadata.level1.parquet").to_pylist() if row["task_id"] in allowed}
    check = scorer()
    labels = {}
    for line in args.results.read_text().splitlines():
        row = json.loads(line)
        if row["benchmark"] != "gaia" or str(row["id"]) not in gold:
            continue
        key = f'gaia:{row["id"]}:{row["path"]}:{str(bool(row["web"])).lower()}'
        answer = row["answer"].strip()
        matches = re.findall(r"(?im)^final answer:\s*(.+?)\s*$", answer)
        if matches:
            answer = matches[-1].strip()
        labels[key] = "correct" if check(answer, gold[row["id"]]) else "incorrect"
    args.output.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"Scored {len(labels)} GAIA responses with official scorer")

if __name__ == "__main__":
    main()
