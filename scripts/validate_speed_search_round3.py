#!/usr/bin/env python3
"""Recompute research findings, retaining the failed output-consistency gate."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path, PurePosixPath

BASE = "5c70e238f6068d373553c48d92786a5cf333d05b"
EXPECTED_DIFFERENCES = {
    ("gemma4b", "explanation", 1),
    ("gemma4b", "learning", 1),
    ("gemma2b", "learning", 1),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def med(rows, key):
    return statistics.median(row[key] for row in rows)


def audit(archive, manifest, source=None):
    require(
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"],
        "archive digest",
    )
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            require(
                member.isfile()
                and not path.is_absolute()
                and ".." not in path.parts
                and member.name not in files,
                "archive member",
            )
            files[member.name] = tar.extractfile(member).read()
    require(set(files) == set(manifest["members"]), "member inventory")
    for name, content in files.items():
        require(
            hashlib.sha256(content).hexdigest() == manifest["members"][name],
            "member digest: " + name,
        )

    def read(name):
        return json.loads(files[name])

    metadata = read("metadata.json")
    require(
        metadata["source_commit"] == BASE and metadata["chip"] == "Apple M4 Pro",
        "environment",
    )
    require(
        sum(d["verified_python_files"] for d in metadata["dependencies"].values())
        == 1616,
        "official dependencies",
    )
    expected_models = read("harness/expected-models.json")
    require(
        set(metadata["models"]) == set(expected_models) and len(expected_models) == 3,
        "model inventory",
    )
    for name, expected in expected_models.items():
        actual = metadata["models"][name]
        require(actual["revision"] == expected["revision"], "model revision")
        for filename, identity in expected["large_files"].items():
            require(actual["files"][filename] == identity, "model weights")
    for name, digest in metadata["harness_sha256"].items():
        require(
            hashlib.sha256(files["harness/" + name]).hexdigest() == digest,
            "harness identity",
        )
    if source:
        for name, digest in metadata["source_hashes"].items():
            require(
                hashlib.sha256((source / name).read_bytes()).hexdigest() == digest,
                "source identity",
            )
    execution = read("execution.json")
    require(execution["released"] and execution["cleanup_confirmed"], "worker release")
    for filename, count in [
        ("chunk-status.json", 10),
        ("tail-status.json", 8),
        ("quality-status.json", 8),
    ]:
        status = read(filename)
        require(
            len(status) == count and all(r["exit"] == 0 for r in status),
            "campaign completion",
        )
        require(
            len({(r["model"], r["variant"], r["trial"]) for r in status}) == count,
            "campaign identities",
        )

    chunk = []
    for row in read("chunk-status.json"):
        name = "raw/chunk-" + row["model"] + "-" + row["trial"] + ".log"
        rows = [
            json.loads(line)
            for line in files[name].decode().splitlines()
            if line.startswith('{"model":')
        ]
        require(len(rows) == 5, "chunk streams")
        require(
            [r["context_target"] for r in rows] == [1024, 4096, 8192, 8192, 4096],
            "chunk contexts",
        )
        require(
            all(
                r["model"] == row["model"]
                and r["variant"] == row["variant"]
                and r["trial"] == row["trial"]
                for r in rows
            ),
            "chunk identity",
        )
        chunk.extend(rows)

    tail_files = sorted(
        n for n in files if n.startswith("http-") and n.endswith(".json")
    )
    require(len(tail_files) == 10, "tail server count")
    tail = []
    for name in tail_files:
        rows = read(name)
        require(
            len(rows) == 4 and [r["turn"] for r in rows] == list(range(4)),
            "tail stream count",
        )
        require(
            [r["usage"]["completion_tokens"] for r in rows] == [32, 512, 64, 64],
            "tail budgets",
        )
        tail.extend(rows)
    quality = []
    for row in read("quality-status.json"):
        name = (
            "quality-"
            + row["model"]
            + "-"
            + row["variant"]
            + "-"
            + row["trial"]
            + ".json"
        )
        rows = read(name)
        require(len(rows) == 8, "quality stream count")
        require(
            {(r["case"], r["turn"]) for r in rows}
            == {
                (c, t)
                for c in ["warmup", "explanation", "code", "learning"]
                for t in [0, 1]
            },
            "quality cases",
        )
        require(
            all(
                r["model"] == row["model"]
                and r["variant"] == row["variant"]
                and r["trial"] == row["trial"]
                for r in rows
            ),
            "quality identity",
        )
        quality.extend(rows)
    diagnostic_rows = read("quality-gemma4b-tail-D-E4.json")
    require(len(diagnostic_rows) == 8, "diagnostic streams")
    all_rows = chunk + tail + quality + diagnostic_rows
    require(len(all_rows) == 162, "complete HTTP inventory")
    for row in all_rows:
        require(
            row["done"]
            and row["finish"] in ("stop", "length")
            and row["usage"]["completion_tokens"] > 0,
            "protocol completion",
        )
        require(
            row["usage"]["total_tokens"]
            == row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"],
            "usage arithmetic",
        )
        require(
            all(math.isfinite(row[k]) and row[k] > 0 for k in ["ttft_s", "decode_s"]),
            "HTTP timing",
        )
        require(
            math.isclose(
                row["decode_tps"],
                (row["usage"]["completion_tokens"] - 1) / row["decode_s"],
                rel_tol=1e-12,
            ),
            "throughput arithmetic",
        )
        row["total_s"] = row["ttft_s"] + row["decode_s"]

    host_files = [n for n in files if n.endswith("-host.json")]
    require(len(host_files) == 29, "host process count")
    samples = []
    for name in host_files:
        host = read(name)
        require(host["exit"] == 0 and len(host["samples"]) >= 3, "clean server exit")
        require(
            all(not s["foreign"] for s in host["samples"]),
            "observed foreign model-sized Python",
        )
        samples.extend(host["samples"])
    instrumentation = [read(n) for n in files if n.endswith("-instrumentation.json")]
    require(len(instrumentation) == 29, "instrumentation inventory")
    for capture in instrumentation:
        if "forward_rows" in capture:
            size = int(capture["variant"])
            require(
                any(int(n) == size for n in capture["forward_rows"]), "chunk activation"
            )
        else:
            events = capture["events"]
            require(
                events[0] == {"cache_mode": "exact", "last_row_both_arms": True},
                "existing prefill optimization/control",
            )
            stores = [e for e in events if "stored" in e]
            require(all(e["aligned"] for e in stores), "snapshot offset alignment")
            if capture["variant"] == "baseline":
                require(not any(e["stored"] for e in stores), "baseline tail modified")
            else:
                require(any(e["stored"] for e in stores), "tail snapshot nonactivation")

    def comparison(rows):
        arms = {v: [r for r in rows if r["variant"] == v] for v in ["baseline", "tail"]}
        require(all(arms.values()), "comparison arms")
        for values in arms.values():
            require(
                len(values) == 2 and len({r["answer_sha256"] for r in values}) == 1,
                "within-arm repeatability",
            )
        same = arms["baseline"][0]["answer_sha256"] == arms["tail"][0]["answer_sha256"]
        baseline, candidate = (
            med(arms["baseline"], "ttft_s"),
            med(arms["tail"], "ttft_s"),
        )
        total = {v: med(values, "total_s") for v, values in arms.items()}
        return {
            "baseline_ttft_s": baseline,
            "candidate_ttft_s": candidate,
            "ttft_reduction_percent": 100 * (1 - candidate / baseline),
            "observed_answer_equal": same,
            "total_s": total,
            "comparable_total_reduction_percent": 100
            * (1 - total["tail"] / total["baseline"])
            if same
            else None,
            "cached_tokens": {
                v: [
                    r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    for r in values
                ]
                for v, values in arms.items()
            },
        }

    code_summary, quality_summary, differences = {}, {}, set()
    for model in ["gemma4b", "gemma2b"]:
        for turn in [1, 2, 3]:
            rows = [
                r
                for r in tail
                if r["model"] == model
                and r["turn"] == turn
                and "pilot" not in r["trial"]
            ]
            result = comparison(rows)
            require(result["observed_answer_equal"], "scripted tail answer changed")
            code_summary[model + "/turn" + str(turn)] = result
        for case in ["explanation", "code", "learning"]:
            for turn in [0, 1]:
                result = comparison(
                    [
                        r
                        for r in quality
                        if (r["model"], r["case"], r["turn"]) == (model, case, turn)
                    ]
                )
                if not result["observed_answer_equal"]:
                    differences.add((model, case, turn))
                quality_summary[model + "/" + case + "/turn" + str(turn)] = result
    require(
        differences == EXPECTED_DIFFERENCES, "output-consistency disposition changed"
    )

    diag = [
        e
        for e in read("gemma4b-tail-D-E4-instrumentation.json")["events"]
        if e.get("diagnostic")
    ]
    require(len(diag) == 1, "clone diagnostic count")
    diag = diag[0]
    require(
        len(diag["clone_array_bits"]) == 48
        and all(diag["clone_array_bits"])
        and diag["clone_resume_logits_bits_equal"],
        "snapshot clone mismatch",
    )
    require(
        all(
            all(m[k] for k in ["same_kind", "offset_equal", "idx_equal"])
            for m in diag["metadata"]
        ),
        "clone metadata",
    )
    require(
        diag["fresh_recompute_finite"] and diag["fresh_recompute_max_abs_logits"] > 0,
        "recompute diagnostic",
    )
    components = {}
    for filename, count in [("quant-prefill.json", 16), ("gdn-prefill.json", 3)]:
        rows = read(filename)
        require(len(rows) == count, "component inventory")
        for row in rows:
            for value in row["numerical"].values():
                checks = value if isinstance(value, list) else [value]
                require(
                    all(
                        c["finite"]
                        and math.isfinite(c["max_abs"])
                        and math.isfinite(c["relative_rms"])
                        for c in checks
                    ),
                    "component numerics",
                )
            for name, values in row["samples_s"].items():
                require(
                    values and all(math.isfinite(v) and v > 0 for v in values),
                    "component timing",
                )
                require(
                    math.isclose(
                        statistics.median(values), row["median_s"][name], rel_tol=1e-12
                    ),
                    "component median",
                )
        components[filename] = [
            {k: v for k, v in row.items() if k != "samples_s"} for row in rows
        ]
    chunk_summary = {}
    for model in ["qwen4b", "gemma4b"]:
        require(
            len(
                {
                    r["answer_sha256"]
                    for r in chunk
                    if r["model"] == model and not r["warmup"]
                }
            )
            == 1,
            "chunk answer changed",
        )
        for target in [4096, 8192]:
            medians = {
                v: med(
                    [
                        r
                        for r in chunk
                        if r["model"] == model
                        and r["context_target"] == target
                        and r["variant"] == v
                    ],
                    "ttft_s",
                )
                for v in ["2048", "512", "1024", "4096"]
            }
            chunk_summary[model + "/" + str(target)] = {
                "median_ttft_s": medians,
                "change_percent": {
                    v: 100 * (1 - t / medians["2048"]) for v, t in medians.items()
                },
            }
    return {
        "production_qualified": False,
        "reason": "Three natural-language follow-up cells change the observed answer; retained as unresolved, not waived.",
        "performance_servers": 28,
        "performance_streams": 154,
        "nonwarm_performance_streams": 118,
        "diagnostic_servers": 1,
        "diagnostic_streams": 8,
        "host_samples": len(samples),
        "swap_bytes": sorted({s["swap"] for s in samples}),
        "scripted_conversation": code_summary,
        "natural_conversation": quality_summary,
        "output_difference_cells": sorted(differences),
        "clone_diagnostic": diag,
        "chunk_screen": chunk_summary,
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.archive, json.loads(args.manifest.read_text()), args.source_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
