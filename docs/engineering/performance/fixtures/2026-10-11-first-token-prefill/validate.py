import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from scripts.benchmark_prefill_first_token import validate_cold

root = Path(sys.argv[1])
d = json.loads((root / "results.json").read_text())
assert d.get("complete") is True
assert len(d["arms"]) == 6 and len(d["rows"]) == 24
assert all(a["exit_code"] in (0, -15) for a in d["arms"])
counts = Counter((r["arm"], r["workload"]) for r in d["rows"])
assert len(counts) == 6 and all(n == 4 for n in counts.values())
for r in d["rows"]:
    for key in ["long_solo", "short_solo", "long", "short"]:
        row = r[key]
        answer = (
            None
            if r["workload"] == "prose"
            else ("7319" if key.startswith("long") else "9283")
        )
        validate_cold(row, answer)
        assert row["done"] and row["finish_reason"] in ("stop", "length")
        assert (
            hashlib.sha256(row["output_text"].encode()).hexdigest()
            == row["output_sha256"]
        )
        assert 0 < row["ttft_ms"] <= row["total_ms"] and math.isfinite(row["total_ms"])
        assert row["usage"]["completion_tokens"] <= (
            96 if r["workload"] == "prose" else 32
        )
    assert r["long"]["usage"]["prompt_tokens"] > 16000
    assert 250 < r["short"]["usage"]["prompt_tokens"] < 400
for a in d["arms"]:
    trace = [
        json.loads(x)
        for x in (root / (a["label"] + ".trace.jsonl")).read_text().splitlines()
    ]
    assert trace and all(0 < r["dispatched"] <= r["selected"] for r in trace)
    shrunk = [r for r in trace if r["dispatched"] < r["selected"]]
    assert bool(shrunk) == ("adaptive" in a["label"])
    if shrunk:
        assert any(r["decoding"] == 0 for r in shrunk)
print(
    "PASS: 24 pairs, 6 clean child exits, protocol/cold cache/output hashes, exact retrieval, bounded dispatch traces"
)
print(json.dumps(d["summary"], indent=2))
for workload in ["prose", "retrieval"]:
    a = d["summary"]["fixed2048"][workload]
    b = d["summary"]["adaptive2048"][workload]
    print(
        workload,
        "short improvement",
        round(100 * (1 - b["short_ttft_ms"] / a["short_ttft_ms"]), 1),
        "long time change",
        round(100 * (b["long_total_ms"] / a["long_total_ms"] - 1), 1),
    )
    for arm in ["fixed2048", "fixed512", "adaptive2048"]:
        rows = [r for r in d["rows"] if r["arm"] == arm and r["workload"] == workload]
        print(
            arm,
            "TTFT samples",
            [round(r["short"]["ttft_ms"]) for r in rows],
            "short sameoutput",
            sum(
                r["short"]["output_sha256"] == r["short_solo"]["output_sha256"]
                for r in rows
            ),
            "long sameoutput",
            sum(
                r["long"]["output_sha256"] == r["long_solo"]["output_sha256"]
                for r in rows
            ),
        )
