#!/usr/bin/env python3
"""Audit retained round-two research receipts without running inference."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path, PurePosixPath

BASE = "5c70e238f6068d373553c48d92786a5cf333d05b"
REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"
WEIGHT = "5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db"
ANSWER = "bfb91d55d5b34a8305d2c1eee06fc1c74ef3739d56ad82fdbac8390020466d7a"
WARM = "30376453205f5a2a96a92c56f5404df9ffe6e1d5ca84938a59a5ed3e7cad9529"
SHORT = [1024] * 3
LONG = [1024, 1024, 8192, 16384, 16384, 8192, 1024]
GROUPS = {
    "A-short": (
        "A",
        [("baseline", "A1"), ("gdn", "B1"), ("gdn", "B2"), ("baseline", "A2")],
        SHORT,
    ),
    "A-pack": (
        "A",
        [("gdn", "P-A1"), ("gdn_pack", "P-B1"), ("gdn_pack", "P-B2"), ("gdn", "P-A2")],
        SHORT,
    ),
    "A-long": (
        "A",
        [("baseline", "L-A1"), ("gdn", "L-B1"), ("gdn", "L-B2"), ("baseline", "L-A2")],
        LONG,
    ),
    "B-long": (
        "B",
        [("baseline", "B-A1"), ("gdn", "B-B1"), ("gdn", "B-B2"), ("baseline", "B-A2")],
        LONG,
    ),
    "A-activation": (
        "A",
        [("gdn", "activation"), ("gdn_pack", "packed_activation")],
        SHORT,
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(archive, manifest, source_dir=None):
    raw = archive.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == manifest["archive_sha256"], "archive digest"
    )
    members = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            require(
                member.isfile() and not path.is_absolute() and ".." not in path.parts,
                "unsafe member",
            )
            require(member.name not in members, "duplicate member")
            members[member.name] = tar.extractfile(member).read()
    require(set(members) == set(manifest["members"]), "member inventory")
    for name, content in members.items():
        require(
            hashlib.sha256(content).hexdigest() == manifest["members"][name],
            "member digest: " + name,
        )

    def read(name):
        return json.loads(members[name])

    execution = read("execution.json")
    require(
        execution["A"]["machine_id"] != execution["B"]["machine_id"],
        "independent workers",
    )
    require(
        all(
            execution[w]["cleanup_confirmed"] and execution[w]["released"]
            for w in ["A", "B"]
        ),
        "worker cleanup",
    )
    for worker in ("A", "B"):
        meta = read(worker + "/metadata.json")
        require(meta["source_commit"] == BASE, "source revision")
        require(
            meta["chip"] == "Apple M4 Pro" and meta["memory_bytes"] == 48 * 1024**3,
            "hardware",
        )
        require(
            sum(d["verified_python_files"] for d in meta["dependencies"].values())
            == 1616,
            "dependency inventory",
        )
        model = meta["models"]["mlx-community/Qwen3.5-4B-MLX-4bit"]
        require(
            model["revision"] == REVISION
            and model["files"]["model.safetensors"]["sha256"] == WEIGHT,
            "weights",
        )
        for name, digest in meta["harness_sha256"].items():
            require(
                hashlib.sha256(members["harness/" + name]).hexdigest() == digest,
                "harness: " + name,
            )
        if source_dir:
            for name, digest in meta["source_hashes"].items():
                require(
                    hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                    == digest,
                    "source: " + name,
                )
        parity = read(worker + "/real-layer-parity.json")
        rows = parity["records"]
        require(parity["enrolled"] == 24 and len(rows) == 192, "learned layer count")
        identities = {(row["module"], row["step"]) for row in rows}
        modules = {row["module"] for row in rows}
        require(
            len(modules) == 24
            and identities == {(name, step) for name in modules for step in range(8)},
            "learned layer identities",
        )
        require(parity["runtime"]["kernel"] == 192, "learned layer activation")
        for row in rows:
            require(
                row["exact_output_conv_recurrent"] == [True] * 3
                and row["max_abs"] == [0.0] * 3,
                "learned layer mismatch",
            )
            require(
                row["cache_dtypes"] == ["mlx.core.bfloat16", "mlx.core.float32"],
                "cache precision",
            )

    require(
        read("A/metadata.json")["source_hashes"]
        == read("B/metadata.json")["source_hashes"],
        "worker source agreement",
    )
    require(
        read("A/metadata.json")["versions"] == read("B/metadata.json")["versions"],
        "worker runtime agreement",
    )
    results = {}
    streams = measured = samples = 0
    swap = {}
    seen = set()
    for group, (worker, runs, targets) in GROUPS.items():
        measurements = {}
        for variant, trial in runs:
            prefix = worker + "/"
            name = prefix + "http-" + variant + "-" + trial + ".json"
            seen.add(name)
            rows = read(name)
            require(len(rows) == len(targets), "stream count: " + name)
            instrumentation = read(
                prefix + "qwen4b-" + variant + "-" + trial + "-instrumentation.json"
            )
            host = read(prefix + "qwen4b-" + variant + "-" + trial + "-host.json")
            require(
                host["exit"] == 0 and len(host["samples"]) >= 5,
                "server exit/observation",
            )
            samples += len(host["samples"])
            swap.setdefault(worker, set()).update(
                row["swap"] for row in host["samples"]
            )
            require(
                all(not row["foreign"] for row in host["samples"]),
                "observed foreign model-sized Python",
            )
            if variant == "baseline":
                require(
                    instrumentation["installation"] == []
                    and not instrumentation["gdn"],
                    "baseline modified",
                )
            else:
                require(
                    instrumentation["installation"]
                    == [{"enrolled_layers": 24, "exact_probe_threadgroup_y": 32}],
                    "enrollment",
                )
                calls = 24 * sum(row["usage"]["completion_tokens"] + 1 for row in rows)
                require(
                    instrumentation["gdn"]["kernel"] == calls
                    and instrumentation["gdn"]["eligible"] == calls,
                    "runtime activation",
                )
                require(
                    not any(
                        instrumentation["gdn"].get(k, 0)
                        for k in [
                            "ineligible",
                            "nonempty_mask",
                            "cache_lengths",
                            "cache_left_padding",
                        ]
                    ),
                    "guard rejection",
                )
                if variant == "gdn_pack":
                    require(
                        instrumentation["gdn"]["packed_projection"] == calls,
                        "packed activation",
                    )
                    require(
                        len(instrumentation["packing"][0]) == 24
                        and all(row["packed"] for row in instrumentation["packing"][0]),
                        "packed layers",
                    )
            for index, (row, target) in enumerate(zip(rows, targets)):
                warmup = index == 0
                count = 32 if warmup else 256
                require(
                    (
                        row["variant"],
                        row["trial"],
                        row["index"],
                        row["context_target"],
                        row["warmup"],
                    )
                    == (variant, trial, index, target, warmup),
                    "stream identity",
                )
                require(
                    row["revision"] == REVISION
                    and row["usage"]["prompt_tokens"] == target - 168,
                    "input identity",
                )
                require(
                    row["usage"]["completion_tokens"] == count
                    and row["sse_content_chunks"] == count
                    and row["finish"] == "length"
                    and row["done"],
                    "stream completion",
                )
                require(
                    row["answer_sha256"] == (WARM if warmup else ANSWER),
                    "answer identity",
                )
                require(
                    all(
                        math.isfinite(row[k]) and row[k] > 0
                        for k in ["ttft_s", "decode_s", "decode_tps"]
                    ),
                    "timing",
                )
                require(
                    math.isclose(
                        row["decode_tps"], (count - 1) / row["decode_s"], rel_tol=1e-12
                    ),
                    "throughput arithmetic",
                )
                streams += 1
                measured += not warmup
                if not warmup:
                    measurements.setdefault(target, {}).setdefault(variant, []).append(
                        row["decode_tps"]
                    )
        results[group] = {}
        for target, arms in measurements.items():
            medians = {arm: statistics.median(values) for arm, values in arms.items()}
            entry = {
                "median_tps": medians,
                "samples_per_arm": {arm: len(values) for arm, values in arms.items()},
            }
            if len(medians) == 2 and group != "A-activation":
                control, candidate = (
                    ("gdn", "gdn_pack") if group == "A-pack" else ("baseline", "gdn")
                )
                entry["throughput_change_percent"] = 100 * (
                    medians[candidate] / medians[control] - 1
                )
            results[group][str(target - 168)] = entry
    require(
        seen == {n for n in members if n.endswith(".json") and "/http-" in n},
        "HTTP inventory",
    )
    for filename, count in [
        ("norm-probe.json", 8),
        ("norm-chain.json", 4),
        ("post-norm-probe.json", 4),
        ("gdn-components.json", 2),
    ]:
        rows = read("A/" + filename)
        require(len(rows) == count, "component inventory")
        for row in rows:
            checks = row.get("exact", row.get("all_exact"))
            require(
                all(checks.values()) if isinstance(checks, dict) else checks is True,
                "component equality",
            )
            if "checks" in row:
                require(
                    {c["seed"] for c in row["checks"]} == set(range(8))
                    and len(row["checks"]) == 8,
                    "component seeds",
                )
                require(
                    all(c["exact"] for c in row["checks"]), "component input equality"
                )
            if "chain_exact" in row:
                require(all(row["chain_exact"].values()), "component chain equality")
            for arm, values in row["samples_s"].items():
                require(
                    all(math.isfinite(v) and v > 0 for v in values), "component timing"
                )
                require(
                    math.isclose(
                        statistics.median(values), row["median_s"][arm], rel_tol=1e-12
                    ),
                    "component median",
                )
    probes = read("A/gdn-probe.json")
    require(
        len(probes) == 8
        and {(r["value_heads"], r["threadgroup_y"]) for r in probes}
        == {(heads, tg) for heads in [32, 48] for tg in [32, 16, 8, 4]}
        and all(
            r["eight_step_output_conv_fp32_state_exact"] and r["error"] is None
            for r in probes
        ),
        "geometry probes",
    )
    require(
        streams == 86 and measured == 68 and len(seen) == 18, "complete campaign counts"
    )
    return {
        "source_commit": BASE,
        "servers": len(seen),
        "streams": streams,
        "measured_streams": measured,
        "host_samples": samples,
        "observed_swap_bytes": {w: sorted(v) for w, v in swap.items()},
        "campaigns": results,
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
