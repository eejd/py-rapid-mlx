#!/usr/bin/env python3
"""Create blinded, position-swapped Arena pairs and factuality review sheets."""
import argparse
import hashlib
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--arena-out", type=pathlib.Path, required=True)
    parser.add_argument("--facts-out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    indexed = {(r["benchmark"], str(r["id"]), r["path"], bool(r.get("web"))): r for r in rows}
    pairs = []
    facts = []
    for (benchmark, id_, path, web), row in indexed.items():
        if benchmark == "arena" and path == "goose":
            direct = indexed.get((benchmark, id_, "direct", False))
            if direct:
                pairs.append({"key": f"arena:{id_}:ab", "first": row["answer"], "second": direct["answer"],
                              "instruction": "Choose first, second, or tie for the more helpful accurate answer; judge only answer content."})
                pairs.append({"key": f"arena:{id_}:ba", "first": direct["answer"], "second": row["answer"],
                              "instruction": "Choose first, second, or tie for the more helpful accurate answer; judge only answer content."})
        if benchmark in ("simpleqa", "chinese"):
            facts.append({"key": f"{benchmark}:{id_}:{path}:{str(web).lower()}", "answer": row["answer"],
                          "instruction": "Label correct, incorrect, or not-attempted against the cached source gold answer. Check aliases and partial answers carefully."})
    # Stable shuffled order avoids grouping swapped pairs next to each other.
    pairs.sort(key=lambda row: hashlib.sha256(("review-order-20261015:" + row["key"]).encode()).hexdigest())
    facts.sort(key=lambda row: row["key"])
    args.arena_out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
    args.facts_out.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(pairs)} blinded Arena comparisons; {len(facts)} factuality answers")

if __name__ == "__main__":
    main()
