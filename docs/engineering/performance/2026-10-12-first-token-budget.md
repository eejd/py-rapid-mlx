# First-token prefill budgets and segment-tail work

Date: 2026-10-12. Owner: Vector. This continues the
[first-token interference study](2026-10-11-first-token-prefill.md). Serving
library code and operator defaults remain unchanged. Both candidates run only
inside the benchmark's owned loopback subprocesses.

## Compute-budget comparison

Question: does widening the pre-first-token compute budget from 500 to 1000 ms
avoid enough small prompt forwards to improve long completion without losing
interactive latency?

Source: `58cb67370`. The harness accepts an explicit arm list, records the
actual target in every dispatch trace, rejects duplicate arms/workloads, and
refuses to mark a campaign complete after an unclean child shutdown. Existing
three-arm invocations retain their previous defaults.

Each arm uses a fresh process in forward and reverse order, with two paired
trials per process: four samples per arm, 16 paired waves and eight child
servers. Every trial measures a short request alone before starting a long
request and then dispatching the same short request once the long one is
observed running. Each process also measures one long request alone, so
long-solo medians have two independent samples per arm. Prefix data is cleared
between waves; model compilation and online cost estimates remain warm.

All requests use the same deterministic retrieval workload: about 16K/256
input tokens plus a four-digit code inserted halfway through the text. Long
and short requests have different codes, 7319 and 9283. Thinking and speculation
are off, temperature is zero, and the output budget is 32. Every measured stream
must stop normally with the exact expected code, positive usage, `[DONE]`, and
zero reused tokens. This is a limited context-retention and row-separation
check, not a general coding or quality evaluation.

Apple M3 Ultra, 256 GiB, macOS 26.5.2; Python 3.12.14, MLX 0.32.3,
mlx-lm 0.31.3 and Transformers 5.15.1. Exact versions are recorded in results.
Existing Qwen3.5-4B 4-bit snapshot
`32f3e8ecf65426fc3306969496342d504bfa13f3`; no model downloads. Another
training job remains active. These are shared-host measurements; they must not
be compared directly to the previous day's absolute timings.

All 16 waves and eight child shutdowns passed validation, including 56 distinct
measured streams and eight warmups. Table values are medians.

| Arm | Short TTFT | Long completion | Long alone completion |
| --- | ---: | ---: | ---: |
| Fixed 2048 | 8.978 s | 17.218 s | 11.778 s |
| Fixed 512 | 2.622 s | 16.354 s | 16.363 s |
| Cost-aware 2048, 500 ms | 2.888 s | 14.793 s | 13.163 s |
| Cost-aware 2048, 1000 ms | 3.508 s | 14.597 s | 12.115 s |

The 1000 ms budget makes short TTFT 21.5% worse than 500 ms while changing long
completion by only -1.3%. The latter is smaller than observed shared-host and
solo-baseline variation. This does not justify widening the budget. The 500 ms
candidate lowers short TTFT by 67.8% relative to fixed 2048 in this campaign.
No claim of a universal long-request acceleration follows from these medians.

## Segment-tail candidate

A different mechanism avoids work rather than allowing a larger compute budget.
The existing prompt runtime right-pads a short row to the longest row in the
same forward. When a short prompt segment has only tens of tokens left, a
long neighbour can still take the full cost-selected chunk. That completes the
short segment but computes padding over much of its row.

Source: `0449ead1d`. After applying the same 500 ms cost bound and checkpoint
grid limit, the new arm caps the chunk at the shortest current prompt segment.
It can only shrink the existing bound. Only rows that can join this tick count;
a final one-token row awaiting promotion is excluded, so it cannot force the
remaining long row into one-token prefill. Single-prompt dispatch is unchanged.
Two focused contracts cover finishing tails, queued admission and promotion;
57 tests across the harness and existing scheduler policies pass.

A forward/reverse pilot uses both fixed controls, the 500 ms cost control and
the tail candidate, with one paired wave per process: two samples per arm.
This is a smaller exploratory pilot, not default qualification.

All eight waves and child shutdowns passed, including 32 distinct measured
streams and eight warmups. Medians:

| Arm | Short TTFT | Long completion | Long alone completion |
| --- | ---: | ---: | ---: |
| Fixed 2048 | 9.431 s | 18.153 s | 12.142 s |
| Fixed 512 | 2.847 s | 16.858 s | 13.741 s |
| Cost-aware 2048, 500 ms | 2.663 s | 14.442 s | 12.013 s |
| Cost-aware plus tail cap | 2.933 s | 15.327 s | 14.072 s |

The tail cap did engage: both paired traces dispatch a 14-token finishing
segment instead of the control's 256/288-token chunks at the same short-row
progress (279 tokens). However, its short TTFT is 10.1% worse and long completion
6.1% worse at the median than the cost control. Individual short TTFTs are
2.486/3.379 s for the tail cap and 2.730/2.595 s for the control: the direction
reverses between cycles. In the reverse cycle, the already-running initial long
chunk takes 1.963 s versus the control's 1.138 s. Short-solo medians also vary
(0.825 versus 0.351 s). These observations prevent attributing the differences
to the candidate alone.

Reducing padding is demonstrated; improving interactive latency is not. The
candidate adds small forwards and remains benchmark-only. Do not adopt it as a
serving default from this pilot.

## Interpretation and remaining work

The initial uncontended chunk still cannot be interrupted after a peer arrives.
Earlier first-token delivery also activates normal decode protection earlier
in the long prompt, so latency gains can affect long completion. The previous
study's +13.2% retrieval long-completion result remains part of the evidence;
this new campaign does not establish its cause or erase it.

Only one small dense hybrid model and one text streaming route are covered.
Different batch shapes can still produce different free-form outputs, as the
previous study records. Exact retrieval agreement does not prove logits or
arbitrary answer equivalence. Tool calls, cancellation histories, high-
concurrency coalescing, saturated admission, other model families and other
chips need separate coverage before a production design or default decision.

Vector owns a quiet-host matched-workload qualification of any retained
candidate, including long completion and prefix-cache histories. Atlas owns a
later default/compatibility disposition; Harbor can schedule an exclusive run.
No production release is part of this experiment.

## Reproduction and evidence

Use the recorded source/runtime and existing local snapshot. Output files must
not already exist. An unused loopback port avoids collisions or recently closed
TCP sockets. The harness stops only its own child processes.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1 \
python scripts/benchmark_prefill_first_token.py \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3" \
  --output /private/tmp/rapid-mlx-prefill-budget/results.json \
  --port 18348 --repeat 2 --cycles 2 --workloads retrieval \
  --arms fixed2048 fixed512 adaptive2048 adaptive2048-1000ms
```

For the tail pilot, use a new output path and port 18349, `--repeat 1`, and
replace the final arm with `adaptive2048-tail`. The current harness includes
both candidates; the archived source/hash identifies the exact measured version.

[Archived evidence](fixtures/2026-10-12-first-token-budget/) includes raw
scheduler traces, normalized client results, environment, validation script and
SHA-256 manifest. Absolute machine paths in commands are normalized, with
original command/result hashes retained. Trace bytes are unchanged.

```bash
python -m pytest tests/test_benchmark_prefill_first_token.py \
  tests/test_adaptive_prefill.py tests/test_scheduler_admission_policy.py -q
PYTHONPATH=. python docs/engineering/performance/fixtures/2026-10-12-first-token-budget/validate.py \
  docs/engineering/performance/fixtures/2026-10-12-first-token-budget/budget \
  fixed2048 fixed512 adaptive2048 adaptive2048-1000ms
PYTHONPATH=. python docs/engineering/performance/fixtures/2026-10-12-first-token-budget/validate.py \
  docs/engineering/performance/fixtures/2026-10-12-first-token-budget/tail \
  fixed2048 fixed512 adaptive2048 adaptive2048-tail
```
