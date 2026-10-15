"""Verify archived long-context decode receipts and recompute comparisons."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import tarfile
from pathlib import Path

import numpy as np


def validate(
    archive: Path, manifest_path: Path, source_dir: Path | None = None
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert all(m.isfile() and m.size < 10 * 1024**2 for m in members)
        assert {m.name for m in members} == set(manifest["files"])
        assert len(members) == len(manifest["files"])
        payloads = {m.name: bundle.extractfile(m).read() for m in members}
    for name, payload in payloads.items():
        assert hashlib.sha256(payload).hexdigest() == manifest["files"][name]

    def read(name):
        return json.loads(payloads["data/" + name])

    phases = {
        "pilot": ["http-baseline.json"],
        "fused": [
            "http-baseline-A1.json",
            "http-d512-B1.json",
            "http-d512-B2.json",
            "http-baseline-A2.json",
        ],
        "mask": [
            "http-baseline-M-A1.json",
            "http-d512_nomask-M-B1.json",
            "http-d512_nomask-M-B2.json",
            "http-baseline-M-A2.json",
        ],
    }
    result = {"archive_members": len(payloads), "phases": {}}
    answers = {}
    for phase, names in phases.items():
        rows = []
        for name in names:
            group = read(name)
            expected = 14 if phase == "pilot" else 7 if phase == "fused" else 3
            assert len(group) == expected, (name, len(group))
            for row in group:
                assert row["done"] and row["finish"] == "length"
                assert row["usage"]["completion_tokens"] == (
                    32 if row["warmup"] else 256
                )
                assert row["decode_s"] > 0 and row["ttft_s"] > 0
                assert np.isclose(
                    row["decode_tps"],
                    (row["usage"]["completion_tokens"] - 1) / row["decode_s"],
                )
                assert "def square_" in row["answer_prefix"]
                if not row["warmup"]:
                    key = (row["model"], row["context_target"])
                    answers.setdefault(key, set()).add(row["answer_sha256"])
            rows.extend(group)
        measured = [r for r in rows if not r["warmup"]]
        comparisons = []
        for model, context in sorted(
            {(r["model"], r["context_target"]) for r in measured}
        ):
            selection = [
                r
                for r in measured
                if (r["model"], r["context_target"]) == (model, context)
            ]
            assert len({r["usage"]["prompt_tokens"] for r in selection}) == 1
            variants = sorted({r["variant"] for r in selection})
            medians = {
                v: statistics.median(
                    r["decode_tps"] for r in selection if r["variant"] == v
                )
                for v in variants
            }
            comparison = {
                "model": model,
                "context_target": context,
                "actual_prompt_tokens": selection[0]["usage"]["prompt_tokens"],
                "median_decode_tps": medians,
                "counts": {
                    v: sum(r["variant"] == v for r in selection) for v in variants
                },
            }
            if len(variants) == 2:
                candidate = next(v for v in variants if v != "baseline")
                comparison["throughput_change_pct"] = (
                    medians[candidate] / medians["baseline"] - 1
                ) * 100
            comparisons.append(comparison)
        result["phases"][phase] = {
            "streams": len(rows),
            "measured": len(measured),
            "comparisons": comparisons,
        }
    assert all(len(values) == 1 for values in answers.values())
    result["matching_observed_output_hashes"] = True

    host = [
        json.loads(value)
        for name, value in payloads.items()
        if name.endswith("-host.json")
    ]
    assert len(host) == 10 and all(h["exit"] == 0 and h["samples"] for h in host)
    samples = [s for h in host for s in h["samples"]]
    result["host"] = {
        "servers": len(host),
        "samples": len(samples),
        "swap_min": min(s["swap"] for s in samples),
        "swap_max": max(s["swap"] for s in samples),
        "foreign_large_python_pids": sorted(
            {p["pid"] for s in samples for p in s["foreign"]}
        ),
    }
    for trial in ["M-A1", "M-B1", "M-B2", "M-A2"]:
        variant = "d512_nomask" if "-B" in trial else "baseline"
        item = read(f"gemma4b-{variant}-{trial}-instrumentation.json")
        assert item["mask_checks"]
        assert all(x["all_true"] for x in item["mask_checks"].values())
        if variant == "d512_nomask":
            assert item["stats"]["mask_removed_calls"] > 0
    errors = []
    for name, value in payloads.items():
        if name.startswith("data/reference-baseline-"):
            a = np.load(io.BytesIO(value), allow_pickle=False)
            b = np.load(
                io.BytesIO(
                    payloads[name.replace("reference-baseline-", "reference-d512-")]
                ),
                allow_pickle=False,
            )
            assert np.isfinite(b).all() and np.allclose(a, b, atol=0.002, rtol=0.002)
            errors.append(float(np.max(np.abs(a - b))))
    assert len(errors) == 12
    result["attention_reference_pairs"] = len(errors)
    result["attention_max_abs_difference"] = max(errors)
    growth = read("cache-growth-components.json")
    for key in {(r["heads"], r["dimension"], r["context"]) for r in growth}:
        group = [r for r in growth if (r["heads"], r["dimension"], r["context"]) == key]
        assert len(group) == 8
        assert len({r["valid_keys_sha256"] for r in group}) == 1
    result["cache_growth_runs"] = len(growth)
    metadata = read("metadata.json")
    assert metadata["source_commit"] == manifest["source_commit"]
    for name, digest in metadata["harness_sha256"].items():
        assert hashlib.sha256(payloads["scripts/" + name]).hexdigest() == digest
    if source_dir is not None:
        for name, digest in metadata["source_hashes"].items():
            assert (
                hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == digest
            )
    result["verified_dependency_python_files"] = sum(
        x["verified_python_files"] for x in metadata["dependencies"].values()
    )
    assert result["verified_dependency_python_files"] == 1616
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.archive, args.manifest, args.source_dir), indent=2))
