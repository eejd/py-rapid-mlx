import hashlib
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from scripts.benchmark_prefill_first_token import validate_cold

root = Path(sys.argv[1])
d = json.loads((root / "results.json").read_text())
expected = set(sys.argv[2:])
assert d.get("complete") is True
assert len(d["arms"]) == len(expected) * d["cycles"]
assert len(d["rows"]) == len(expected) * d["cycles"] * d["repeat"] * len(d["workloads"])
assert all(a["exit_code"] in (0, -15) for a in d["arms"])
counts = Counter((r["arm"], r["workload"]) for r in d["rows"])
assert set(a for a, w in counts) == expected
assert all(n == d["cycles"] * d["repeat"] for n in counts.values())
for r in d["rows"]:
    assert r["workload"] == "retrieval"
    for k in ["long_solo", "short_solo", "long", "short"]:
        row = r[k]
        validate_cold(row, "7319" if k.startswith("long") else "9283")
        assert (
            row["done"]
            and 0 < row["ttft_ms"] <= row["total_ms"]
            and math.isfinite(row["total_ms"])
        )
        assert (
            row["output_sha256"]
            == hashlib.sha256(row["output_text"].encode()).hexdigest()
        )
        assert row["usage"]["completion_tokens"] <= 32
for arm in expected:
    rows = [r for r in d["rows"] if r["arm"] == arm]
    for key, who, metric in [
        ("short_ttft_ms", "short", "ttft_ms"),
        ("short_solo_ttft_ms", "short_solo", "ttft_ms"),
        ("long_total_ms", "long", "total_ms"),
        ("long_solo_total_ms", "long_solo", "total_ms"),
    ]:
        assert math.isclose(
            d["summary"][arm]["retrieval"][key], st.median(r[who][metric] for r in rows)
        )
for a in d["arms"]:
    traces = [
        json.loads(x)
        for x in (root / (a["label"] + ".trace.jsonl")).read_text().splitlines()
    ]
    assert traces and all(0 < t["dispatched"] <= t["selected"] for t in traces)
    assert all(t["target_ms"] == a["effective_target_ms"] for t in traces)
    assert a["effective_target_ms"] == (
        1000 if "1000ms" in a["label"] else d["target_ms"]
    )
    if "-tail" in a["label"]:
        assert any(
            t["decoding"] == 0
            and t["dispatched"] == 14
            and any(r["segments"][0] == 14 for r in t["processing"])
            for t in traces
        )
    assert any(
        t["dispatched"] < t["selected"] and t["decoding"] == 0 for t in traces
    ) == ("adaptive" in a["label"])
print(
    "PASS:",
    len(d["rows"]),
    "paired waves;",
    len(d["arms"]),
    "clean server arms; cold streams, exact answers, hashes, recomputed medians and trace targets",
)
print(json.dumps(d["summary"], indent=2))
