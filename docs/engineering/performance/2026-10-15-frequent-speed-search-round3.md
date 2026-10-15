# Round three: large follow-up cache gains, unresolved output consistency

This round applies a greater-than-10% latency-reduction threshold to frequent
workloads. No new production change qualifies. Three computation candidates
fall below the threshold or lose performance. Retaining a completed text turn's
cache produces much larger follow-up gains, including unchanged-answer cases,
but broader testing finds reproducible output differences. That finding remains
open; this report does not waive it or propose a serving PR.

## What helps a common conversation

The existing multimodal text lane retains prompt-boundary snapshots. Its
completed response also has usable computed cache state, including the tokens
just generated. The private prototype adds that completed-turn snapshot to the
existing exact-prefix cache, retaining its token-prefix check, semantic identity,
LRU and byte budget. The next ordinary follow-up can then avoid recomputing the
previous answer. No concurrent requests, speculation, draft model, changed chat
template or replacement token IDs are required.

Both comparison arms have prefix caching enabled and disk caches disabled.
Both also apply the previously investigated last-position logits optimization.
Consequently these results do not rediscover that earlier prefill saving.
The prototype only accepts text-only cases with an exact cache manager and
checks every cache offset against the complete raw token sequence before storing.
The compute-ahead pipeline has already consumed the emitted token. Raw control
EOS is included in the aligned state identity, even though public completion
usage excludes that control token.

Four fresh processes per model run baseline/candidate/candidate/baseline.
Each performs a 32-token warmup, a 512-token numbered-code response, and two
64-token follow-ups. All observed answers match across comparison arms and
repeat within each arm. These are narrow scripted conversations, not broad
quality qualification.

| Model and request | Baseline TTFT | Candidate TTFT | TTFT reduction | Complete-response reduction |
| --- | ---: | ---: | ---: | ---: |
| Gemma E4B: first follow-up after 512 output tokens | 580.00 ms | 105.45 ms | 81.82% | 29.97% |
| Gemma E2B: first follow-up after 512 output tokens | 179.06 ms | 49.40 ms | 72.41% | 17.33% |
| Gemma E4B: next follow-up after 64 output tokens | 172.68 ms | 105.31 ms | 39.01% | 5.36% |
| Gemma E2B: next follow-up after 64 output tokens | 68.98 ms | 49.96 ms | 27.57% | 3.05% |

The 512-token response's total time changes by less than 0.1% on either model.
It reuses the warmup's input prefix in both arms, so this is a steady-state
control rather than cold-chat qualification.
For the E4B first follow-up, reusable prefix tokens increase from 843 to 1,366;
the full next prompt is 1,386 tokens. This establishes actual additional reuse,
not merely a snapshot counter. The shorter follow-up shows why a large TTFT
percentage is not automatically a greater-than-10% total-response benefit.

## Broader conversations do not pass the output gate

A second four-process campaign per model tests three ordinary topics:
an explanation and summary, Python CSV code and edge cases, and an SQL learning
plan and revision. Initial budgets are 512/512/384 tokens; follow-ups allow 96.
Natural EOS and length finishes are both accepted. This version also attempts
an aligned snapshot after natural EOS. Every initial answer matches across
arms, and each arm repeats its own answers consistently.

| Follow-up cell | Observed answers equal across arms? | Additional cache reuse? | Disposition |
| --- | --- | --- | --- |
| E4B explanation | No | Yes | Output-consistency gate unresolved; no complete-response speed claim. |
| E4B code | Yes | Yes | TTFT 563.5 to 102.0 ms; complete response 1.304 to 0.839 s, about 35.6% shorter. |
| E4B learning plan | No | Yes | Output-consistency gate unresolved. |
| E2B explanation | Yes | Yes | TTFT 120.5 to 43.5 ms; complete response 0.613 to 0.526 s, about 14.1% shorter. |
| E2B code | Yes | No | Both reuse only 25 tokens and take approximately 168 ms to first text; no gain. |
| E2B learning plan | No | Yes | Output-consistency gate unresolved. |

Three of the six model/topic cells change the observed answer. The baseline
and candidate each reproduce their own different answer on both processes;
these are not isolated timing outliers. No semantic-quality judgment or
accuracy benchmark was used to excuse the differences. The E2B code case also
shows that storing a snapshot does not guarantee its later reuse. The exact
reason for that miss was not isolated, and the prototype does not rewrite
inputs to force a hit.

A separate diagnostic samples an actual 318-token E4B completed-turn snapshot.
All 48 live/stored cache arrays agree by storage bits, as do cache types,
offsets and rotating-buffer indices. Copies of the live and stored caches,
resumed on the same three-token diagnostic suffix `[7, 8, 9]`, produce bitwise
identical logits. Independently rebuilding the same token prefix in a fresh
cache and then processing that suffix gives maximum absolute logit difference
0.859375 and relative RMS difference 0.0346109; values remain finite and this
single suffix's final argmax agrees.

This rules out a clone mismatch in the sampled diagnostic. It does not identify
every cause of the natural-language differences, bound model-quality impact,
or establish that all cached and replayed trajectories are interchangeable.
The eight diagnostic requests include extra model work and are excluded from
performance comparisons. No other hardware campaign was launched while this
qualification gate remained open.

## Other candidates stopped early

1. **Dequantize then use dense prefill matmul:** sixteen synthetic cases use
   input/output widths 2,560/10,240, 10,240/2,560, 2,560/2,560 and 5,120/17,408,
   with 1/128/512/2,048 rows. At 2,048 rows, materializing the dequantized weight
   each call gives approximately 8% component-throughput improvement, less than
   8% component-latency reduction. Keeping the dense weight gives roughly 10%
   component throughput with about four times the packed weight storage before
   counting the original representation. Whole-request latency cannot inherit
   that entire saving, and the single-row path is substantially slower.
   The single-row numerical results also differ. No HTTP promotion.

2. **Change text prefill chunks from 2,048 to 512/1,024/4,096:** ten real HTTP
   processes cover Qwen 4B and Gemma E4B, with two measurements at each of
   approximately 4K and 8K input tokens. Qwen TTFT is essentially unchanged.
   Gemma's 512-token variant saves about 3.5–3.9%, below admission. A 4,096-token
   chunk causes a large short-prompt regression when that prompt takes the
   existing unchunked/full-logits path; it is not a new positive opportunity.
   All measured answers match within each model. These controls use the baseline
   prefill implementation and disable prefix caching.

3. **Use the dependency's parallel GDN prefill scan on Metal:** sequence lengths
   128/512/2,048 and scan chunks 16/32/64 are compared with the existing sequential
   Metal operator. At 2,048 tokens, native takes 6.95 ms versus 14.17 ms for the
   best tested parallel variant. Numerical differences are retained rather than
   called bitwise equivalence. No real-model deployment or HTTP promotion.

The prior 3–4% dense-GDN enrollment lead is not pursued further under the new
threshold. Existing sampler, projection, attention-layout and normalization
negatives are not counted again as fresh discoveries.

## Environment, evidence and reproduction

Baseline `5c70e238f6068d373553c48d92786a5cf333d05b`, one leased Apple M4 Pro
with 48 GiB, Python 3.12.13, MLX 0.32.3, MLX-LM 0.31.3, MLX-VLM 0.7.2,
Transformers 5.15.1, NumPy 2.4.4, HTTPX 0.28.1 and psutil 7.2.2.
The checkout runs through `PYTHONPATH`; inherited Rapid package metadata says
0.15.7 and is not a packaged-release qualification. Twelve relevant source files,
fourteen executed harnesses and 1,616 official dependency Python files are
verified. The receipt binds the immutable snapshots and official large-file
hashes for Qwen 4B (`32f3e8ec...`), Gemma E4B (`475b9088...`) and Gemma E2B
(`23876752...`). No new model downloads occur.

Twenty-eight timing processes complete 154 streams, including 118 non-warmup
observations. The diagnostic adds one process and eight requests separately.
Across 720 host samples, existing swap remains 195,493,888 bytes and no foreign
Python process exceeds the observer's 512 MiB threshold. This finite observation
does not certify GPU isolation or exclude other OS activity. The owned worker
processes and scratch environment are removed and its lease released after
collection; borrowed runtimes and standard model caches remain intact.

The chunk runner initially names its JSON snapshot by variant/trial, so the
second model overwrites the first model's snapshot file. Every individual row
is also printed into a distinct per-model stdout log. The archive retains all
ten complete raw logs as the canonical 50-row record, and the validator derives
the results from those logs. No missing timing was invented or selectively
rerun. The overwritten convenience snapshots remain only in the private raw
receipt archive.

The [evidence archive](fixtures/2026-10-15-speed-search-round3/evidence.tar.gz)
and [manifest](fixtures/2026-10-15-speed-search-round3/manifest.json) include the
exact prototypes, raw observations and provenance. Recompute the
[summary](fixtures/2026-10-15-speed-search-round3/summary.json) with:

```sh
python scripts/validate_speed_search_round3.py \
  docs/engineering/performance/fixtures/2026-10-15-speed-search-round3/evidence.tar.gz \
  docs/engineering/performance/fixtures/2026-10-15-speed-search-round3/manifest.json \
  --source-dir .
```

The audit must return `production_qualified: false` and retain the three exact
output-difference cells. Passing the evidence audit is not passing the quality
gate. For fresh inference, acquire a worker lease, use the pinned runtime and
standard-cache snapshots, create `/private/tmp/rapid-mlx-speed-search-round3`
with `src`, `venv` and `.task-owner` containing `vector-speed-search-round3-v1`,
and extract the archived harness there. Run `driver_chunk.py`, then
`gdn_prefill.py`, then `driver_tail.py` and `driver_quality.py` serially.
The last driver waits for all eight preceding tail runs to finish. Run
`run_tail_diag.py tail gemma4b` with a distinct `STUDY_TRIAL` only after timings;
retain its results separately. Run `metadata.py` after inference so weight
hashing cannot contaminate timings.

There is no cross-chip, Qwen 27B, TensorFold, arbitrary-template, cancellation,
cache-pressure, tool/media or broad quality qualification in this round. Vector
owns the remaining numerical diagnosis and lifecycle work; Atlas must disposition
the product compatibility contract before any production enrollment. A large
conditional speedup alone does not close those requirements.
