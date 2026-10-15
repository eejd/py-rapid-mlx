#!/usr/bin/env python3
"""Conservative summary of per-item Goose QA results.

Input: JSONL records described in README. Optional --labels JSON maps
"benchmark:id:path:web" to a label. Pairwise Arena labels must have both
"ab" and "ba" entries, where ab puts Goose first and ba puts Goose second.
"""
import argparse
import collections
import json
import pathlib
import statistics

FIELDS = ("first_visible_s", "total_s", "tool_calls", "permission_prompts")


def median(records, key):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return round(statistics.median(values), 2) if values else None


def summarize(records, labels):
    groups = collections.defaultdict(list)
    for record in records:
        groups[(record["benchmark"], record["path"], bool(record.get("web")))].append(record)
    output = []
    for (benchmark, path, web), items in sorted(groups.items()):
        row = {"benchmark": benchmark, "path": path, "web": web, "n": len(items),
               "failures": sum(bool(r.get("failure")) for r in items),
               "empties": sum(bool(r.get("empty")) or not str(r.get("answer", "")).strip() for r in items)}
        row.update({f"median_{key}": median(items, key) for key in FIELDS})
        if benchmark in ("simpleqa", "chinese", "gaia", "ifeval", "gdpval"):
            counts = collections.Counter()
            for r in items:
                key = f'{benchmark}:{r["id"]}:{path}:{str(web).lower()}'
                label = labels.get(key) or r.get("score") or "pending"
                counts[label] += 1
            row["scores"] = dict(sorted(counts.items()))
        output.append(row)
    return output


def arena_pairs(records, labels):
    ids = {r["id"] for r in records if r["benchmark"] == "arena"}
    counts = collections.Counter()
    for id_ in ids:
        ab = labels.get(f"arena:{id_}:ab")
        ba = labels.get(f"arena:{id_}:ba")
        if ab not in ("first", "second", "tie") or ba not in ("first", "second", "tie"):
            counts["pending"] += 1
        elif ab == "tie" and ba == "tie":
            counts["tie"] += 1
        elif ab == "first" and ba == "second":
            counts["goose_win"] += 1
        elif ab == "second" and ba == "first":
            counts["direct_win"] += 1
        else:
            counts["position_disagreement"] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--labels", type=pathlib.Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    print(json.dumps({"summary": summarize(records, labels), "arena_pairwise": arena_pairs(records, labels)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
