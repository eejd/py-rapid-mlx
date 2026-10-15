# M4 long-context decode: small execution changes did not qualify

This investigation found no new serving change worth promoting from the
tested candidates. Larger contexts do reduce ordinary single-user decode
throughput, but the tested cache/layout changes either lost performance or
delivered less than 2% in the actual HTTP workload. Production code and
defaults are unchanged.

## Environment and method

Baseline: `5c70e238f6068d373553c48d92786a5cf333d05b`.
Apple M4 Pro, 48 GiB, macOS 26.5.1, Python 3.12.13, MLX 0.32.3,
MLX-LM 0.31.3, MLX-VLM 0.7.2, Transformers 5.15.1, NumPy 2.4.4,
HTTPX 0.28.1 and psutil 7.2.2. The repository ran through `PYTHONPATH`;
inherited distribution metadata says Rapid 0.15.7 and is not a packaged
release qualification. Source hashes were checked against the baseline.
The 1,616 dependency Python files match their recorded official wheel
manifests; model weight hashes match official snapshot metadata.

| Model | Snapshot | Weight SHA-256 |
| --- | --- | --- |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `32f3e8ecf65426fc3306969496342d504bfa13f3` | `5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db` |
| `mlx-community/gemma-4-e4b-it-4bit` | `475b9088d29754a3379866cf5aeb6b41acd313c2` | `932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff` |

Requests used the default HTTP serving route, temperature zero, no speculation,
and disabled prefix/disk caches. A repeated ` data` prefix precedes an instruction
to generate numbered Python functions. Each measured response generates exactly
256 tokens; warmups generate 32. Timing is local request start to first visible
delta for TTFT; decode throughput is `(completion_tokens - 1) / (response_end -
first_visible_delta)`. This includes transport/output handling and is not a
per-kernel rate or an exact per-token latency distribution.

Ten fresh server processes completed 54 streams: 44 measured and 10 warmups.
All streams had the requested token count, `length` finish, SSE terminator and
the expected code prefix. All 44 measured responses retain matching observed output hashes within each
model across variants and context lengths. This is a narrow deterministic workload,
not broad model-quality qualification. Instrumentation is present in both
comparison arms. The initial context-scaling pilot predates disabling telemetry;
both subsequent A/B campaigns disable it and use their own matched controls.

## Baseline context cost

The exploratory pilot has two measurements per model/context:

| Model | Actual input tokens | Median decode tokens/s |
| --- | ---: | ---: |
| Qwen 4B | 856 | 80.17 |
| Qwen 4B | 8,024 | 72.60 |
| Qwen 4B | 16,216 | 65.94 |
| Gemma E4B | 854 | 68.00 |
| Gemma E4B | 8,022 | 62.29 |
| Gemma E4B | 16,214 | 56.46 |

This establishes context-sensitive decode cost; it does not attribute the
whole difference to attention or prove an available optimization.

## Candidate dispositions

1. **Compact live KV before each attention call: rejected.** Synthetic
   component probes use the three observed query/KV head geometries, bf16 and
   sliced capacity buffers at 1K/8K/16K/32K. Copying K/V to contiguous storage
   generally increases component time, reaching approximately twice the native
   cost at long contexts. Do not introduce a full-history copy to improve reads.

2. **Increase KV growth slabs from 256 to 2,048: not promoted.** Forty-eight
   cache-update trials preserve the exact valid-state checksum. For the
   four-KV-head, 256-dimension, 16K case, 512 updates take median 63.86 versus
   60.27 ms. Other geometries are nearly neutral. This small isolated saving
   requires extra capacity and is not an HTTP speedup result.

3. **Enable the existing 512-dimension fused attention path: rejected as a
   serving optimization.** The installed runtime normally excludes the observed
   GQA factor four from this path. The experimental process sets
   `MLX_SDPA_D512_MIN_KL=0`; no custom kernel or precision change is introduced.
   Unmasked synthetic 16K attention falls from 608.46 to 523.17 microseconds,
   about 14%, but the real route supplies a rank-four mask. Four fresh HTTP
   processes run baseline/candidate/candidate/baseline, with four measurements
   per arm/context:

   | Actual Gemma input tokens | Baseline tokens/s | Candidate tokens/s | Change |
   | --- | ---: | ---: | ---: |
   | 854 | 68.04 | 67.48 | −0.82% |
   | 8,022 | 62.13 | 62.01 | −0.20% |
   | 16,214 | 56.71 | 56.78 | +0.12% |

4. **Combine fused attention with removal of a known all-valid mask: not
   promoted.** A second four-process balanced campaign tests only unpadded B=1
   full attention with 512-dimension heads. Both arms verify boolean mask shape
   and that every position is valid at observed context buckets. This is a
   benchmark-only restriction, not a safe general mask-removal rule. Two
   measurements per arm/context produce:

   | Actual Gemma input tokens | Matched baseline tokens/s | Candidate tokens/s | Change |
   | --- | ---: | ---: | ---: |
   | 8,022 | 62.34 | 63.11 | +1.23% |
   | 16,214 | 56.59 | 57.66 | +1.90% |

   This is too little evidence and too small a measured benefit to justify
   broadening production compatibility. Never combine its numbers with the
   preceding campaign's differently instrumented baseline.

The runtime already uses fused vector attention and selects two-pass execution
for qualifying long GQA inputs. Forcing the existing fused path at dimension
256 is a negative control, not a missing optimization.

Twelve saved component output pairs pass finite-value checks and
`allclose(atol=0.002, rtol=0.002)`; maximum absolute difference is 0.00146484375.
This is a component numerical check, not proof of accuracy for arbitrary models.
Across 671 host samples, existing swap stays at 195,493,888 bytes and no foreign
Python process exceeds the observer's 512 MiB threshold. This does not exclude
all operating-system activity or establish universal hardware percentages.

## Reproduction and next boundary

The [evidence archive](fixtures/2026-10-15-m4-decode/evidence.tar.gz) includes
the exact experimental launchers, raw JSON receipts and saved component outputs.
Its [manifest](fixtures/2026-10-15-m4-decode/manifest.json) binds every member;
the [summary](fixtures/2026-10-15-m4-decode/summary.json) is recomputed by:

```sh
python scripts/validate_m4_decode_study.py \
  docs/engineering/performance/fixtures/2026-10-15-m4-decode/evidence.tar.gz \
  docs/engineering/performance/fixtures/2026-10-15-m4-decode/manifest.json \
  --source-dir .
```

For fresh inference reproduction, reserve an otherwise available M4 and use
the pinned baseline/dependencies and existing standard-cache snapshots above.
The archived launchers expect `/private/tmp/rapid-mlx-m4-decode-study`, a `src`
checkout of the baseline, a `venv`, and `.task-owner` containing exactly
`vector-m4-decode-study-v1`. Extract the archived scripts there. Use
`run-http-pilot.py baseline` for context scaling with `serve-pilot.py` installed
as `serve.py`; restore the archived final `serve.py` afterward. Then run
`components.py baseline`
and `components.py d512`. For the first HTTP comparison, run `run_http.py`
with `(variant, STUDY_TRIAL)` values `(baseline,A1)`, `(d512,B1)`, `(d512,B2)`,
`(baseline,A2)` and model argument `gemma4b`. For the second use `run-mask.py`
with `(baseline,M-A1)`, `(d512_nomask,M-B1)`, `(d512_nomask,M-B2)`,
`(baseline,M-A2)`. Each launcher starts and drains its own server; run them
serially. Preserve separate pilot and final launcher filenames when replaying.
For example, the first HTTP A/B campaign is:

```sh
STUDY_TRIAL=A1 venv/bin/python run_http.py baseline gemma4b
STUDY_TRIAL=B1 venv/bin/python run_http.py d512 gemma4b
STUDY_TRIAL=B2 venv/bin/python run_http.py d512 gemma4b
STUDY_TRIAL=A2 venv/bin/python run_http.py baseline gemma4b
```

Vector owns any follow-up. A useful next experiment needs a materially larger
whole-model effect, such as reducing the dominant decode work, with a new
profile and numerical contract. These results do not justify new defaults,
a serving PR, cross-chip claims, Qwen 27B claims, or performance claims for
another backend. They were not measured in combination with #4658.
Task-owned processes, environment and scratch on the worker were removed and
the M4 lease released; standard model snapshots and the borrowed runtime remain.
