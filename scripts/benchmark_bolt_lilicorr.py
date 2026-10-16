# SPDX-License-Identifier: Apache-2.0
"""Offline acceptance on a canonical serial-greedy target trajectory.

Run under the shared large-model lock and an active Machine Control lease.
Every possible anchor is evaluated; round statistics then traverse that frozen
trajectory by accepted_prefix+1, exactly as greedy speculation would advance.
The target never consumes proposed tokens, so no recurrent rollback is required.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

from rapid_mlx.models.lilicorr import LiLiCorr
from rapid_mlx.utils.tokenizer import load_model_with_fallback

PROMPTS = [
    (
        "coding",
        "Write a Python LRU cache using OrderedDict with get and put methods. Explain the invariant and include a short usage example.",
    ),
    (
        "coding",
        "Implement a Python function merging overlapping closed intervals. Handle empty input, duplicate endpoints, and unsorted intervals. Provide tests and complexity.",
    ),
    (
        "math",
        "Solve 3x + 7 = 31, then derive a formula for the sum of the first n odd positive integers and prove it by induction.",
    ),
    (
        "math",
        "A bag holds 5 red and 3 blue balls. Two balls are drawn without replacement. Compute the probability of exactly one red ball and explain each step.",
    ),
    (
        "chat",
        "I have twenty minutes after work and want to start drawing. Suggest a realistic first-week routine with simple exercises and encouragement.",
    ),
    (
        "chat",
        "Explain to a curious twelve-year-old why the Moon has phases and why a lunar eclipse does not happen every month.",
    ),
    (
        "writing",
        "Write a quiet scene of about 250 words where a lighthouse keeper receives a letter addressed to someone who disappeared ten years ago.",
    ),
    (
        "writing",
        "Draft a friendly email announcing a neighborhood seed exchange on Saturday at 10 am in the library garden. Include what to bring and a rain plan.",
    ),
    (
        "rag",
        "Use only these meeting notes to write an accurate summary with decisions, owners, and dates. Notes: The Atlas sensor pilot has 240 units. On March 3, Mina reported 12 failed humidity sensors; battery tests passed on all units. Jo will replace the faulty sensors by March 12. The team approved a 60-unit field trial starting March 18, pending replacements. Ari owns consent forms, due March 14. The budget stays at $18,000. No decision was made about expansion. Do not invent outcomes.",
    ),
    (
        "rag",
        "Answer using the supplied policy and then summarize it. Policy: Standard members may borrow four books for 21 days. Premium members may borrow eight for 35 days. Either tier may renew twice unless another reader has reserved the book. Lost books cost replacement price plus $5. Children under 12 require guardian consent. Question: Can a standard member renew a reserved book, and what would losing a $24 book cost?",
    ),
    (
        "tool",
        "You have tools get_weather(city: string, date: string) and create_reminder(title: string, time: string). Return only JSON with an array named tool_calls containing name and arguments. User: Check the weather in Seattle for 2026-10-20 and remind me to pack a raincoat at 2026-10-19T19:00:00.",
    ),
    (
        "tool",
        "Return a JSON tool call for search_documents(query: string, limit: integer). User: Find the three most relevant internal documents about rotating database credentials. Use the tool exactly once, then explain in one sentence what you are searching for.",
    ),
]


def sync(cache, *arrays):
    mx.eval(*arrays, [c.state for c in cache])


def summarize(prompts):
    groups = [[p["rows"][i] for i in p["round_positions"]] for p in prompts]
    groups = [g for g in groups if g]
    rows = [row for group in groups for row in group]
    if not rows:
        return {"rounds": 0}
    rng = random.Random(42)
    totals = [(sum(r["output_tokens"] for r in group), len(group)) for group in groups]
    boot = []
    for _ in range(5000):
        sample = rng.choices(totals, k=len(totals))
        boot.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    boot.sort()
    return {
        "rounds": len(rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "mean_accepted_drafts": statistics.mean(r["accepted_drafts"] for r in rows),
        "prompt_macro_mean_output_tokens": statistics.mean(s / n for s, n in totals),
        "prompt_cluster_bootstrap_95pct": [boot[125], boot[4874]],
        "accepted_draft_histogram": [
            sum(r["accepted_drafts"] == k for r in rows) for k in range(8)
        ],
        "mean_draft_ms": statistics.mean(r["draft_ms"] for r in rows),
        "median_draft_ms": statistics.median(r["draft_ms"] for r in rows),
        "mean_round_context_update_ms": statistics.mean(
            r["round_context_update_ms"] for r in rows
        ),
    }


def qualify_nvfp4(draft):
    """Actual-export BF16 native arithmetic vs FP32 dequantized reference."""
    checks = []
    mx.random.seed(42)
    for name, weight in draft.tensors.items():
        if not name.endswith(".weight") or weight.dtype != mx.uint32:
            continue
        prefix = name[:-7]
        q = weight[:64]
        scales = draft.tensors[prefix + ".weight_scale"][:64]
        global_scale = draft.tensors[prefix + ".weight_scale_2"]
        x = mx.random.normal((2, q.shape[1] * 8)).astype(mx.bfloat16)
        actual = mx.quantized_matmul(
            x, q, scales, group_size=16, bits=4, mode="nvfp4"
        ) * global_scale.astype(x.dtype)
        dense = mx.dequantize(q, scales, group_size=16, bits=4, mode="nvfp4").astype(
            mx.float32
        )
        reference = (x.astype(mx.float32) @ dense.T) * global_scale
        error = actual.astype(mx.float32) - reference
        relative = mx.sqrt(
            mx.sum(error**2) / mx.maximum(mx.sum(reference**2), 1e-20)
        ).item()
        checks.append(
            {
                "matrix": prefix,
                "relative_l2": relative,
                "max_abs": mx.max(mx.abs(error)).item(),
            }
        )
        if not math.isfinite(relative) or relative > 0.02:
            raise ValueError(f"Actual NVFP4 arithmetic check failed: {checks[-1]}")
    return checks


def measure_verify(target, tokenizer, lengths, draft):
    """T1..T8 from equivalent caches, 12 samples after two warmups."""
    corpus = "\n\n".join(prompt for _, prompt in PROMPTS)
    ids = tokenizer.encode(corpus)
    ids = (ids * (max(lengths) // len(ids) + 2))[: max(lengths) + 128]
    measurements = []
    from mlx_lm.generate import generate_step, generation_stream

    for length in lengths:
        cache = target.make_cache()
        prefill_taps = []
        for offset in range(0, length, 256):
            hidden, features = target.backbone(
                mx.array([ids[offset : min(offset + 256, length)]]),
                cache=cache,
                target_layer_ids=draft.draft["target_layer_ids"],
            )
            y = target.lm_head(hidden[:, -1:])
            sync(cache, y, features)
            prefill_taps.append(features)
        anchor_token = mx.argmax(y[:, -1], axis=-1).item()
        context = draft.append_context(mx.concatenate(prefill_taps, axis=1))
        mx.eval(context.kv, context.anchor)
        draft_samples = []
        for rep in range(14):
            start = time.perf_counter()
            proposal = draft.propose(
                anchor_token, context, target.backbone.embeddings, target.lm_head
            )
            mx.eval(proposal)
            if rep >= 2:
                draft_samples.append((time.perf_counter() - start) * 1000)
        del context, prefill_taps
        samples = {t: [] for t in range(1, 9)}
        for rep in range(14):
            pos = length + rep * 8
            for t in range(1, 9):
                cloned = []
                for c in cache:
                    clone = copy.copy(c)
                    clone.__dict__ = c.__dict__.copy()
                    if hasattr(c, "cache"):
                        clone.cache = list(c.cache)
                    for name, value in list(clone.__dict__.items()):
                        if isinstance(value, mx.array):
                            setattr(clone, name, value + mx.zeros_like(value))
                    cloned.append(clone)
                inputs = mx.array([ids[pos : pos + t]])
                sync(cloned, inputs)
                start = time.perf_counter()
                y = target(inputs, cache=cloned)
                sync(cloned, y)
                if rep >= 2:
                    samples[t].append((time.perf_counter() - start) * 1000)
                del cloned
            y = target(mx.array([ids[pos : pos + 8]]), cache=cache)
            sync(cache, y)
        measurements.append(
            {
                "context": length,
                "real_context_draft_ms_samples": draft_samples,
                "real_context_draft_median_ms": statistics.median(draft_samples),
                "input_ids": ids[: length + 112],
                "rows": [
                    {"T": t, "median_ms": statistics.median(times), "samples_ms": times}
                    for t, times in samples.items()
                ],
            }
        )
        # Ordinary asynchronous greedy decode, with prefill/first token excluded.
        # Fixed token count avoids EOS-length confounding in this cost control.
        pipeline_tps = []
        for rep in range(4):
            generated = generate_step(
                mx.array(ids[:length]), target, max_tokens=257, prefill_step_size=256
            )
            next(generated)
            start = time.perf_counter()
            decoded = [token for token, _ in generated]
            mx.synchronize(generation_stream)
            elapsed = time.perf_counter() - start
            if len(decoded) != 256:
                raise ValueError("Unexpected pipelined decode length")
            if rep:
                pipeline_tps.append(256 / elapsed)
        measurements[-1]["pipelined_baseline_tps_samples"] = pipeline_tps
        measurements[-1]["pipelined_baseline_median_tps"] = statistics.median(
            pipeline_tps
        )
        print(
            "verify",
            length,
            [(t, round(statistics.median(v), 2)) for t, v in samples.items()],
            flush=True,
        )
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--prompt-limit", type=int, default=len(PROMPTS))
    parser.add_argument("--verify-contexts", nargs="+", type=int, default=[1024, 8192])
    args = parser.parse_args()
    if args.tokens < 16:
        parser.error("--tokens must be at least 16")
    print("Loading", args.target, flush=True)
    target, tokenizer = load_model_with_fallback(args.target)
    draft = LiLiCorr.from_pretrained(args.drafter)
    mx.eval(target.parameters(), draft.parameters())
    taps = draft.draft["target_layer_ids"]
    result = {
        "target": args.target,
        "drafter": args.drafter,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [
                "rapid_mlx/models/lilicorr.py",
                "rapid_mlx/models/nemotron_h_puzzle.py",
                __file__,
            ]
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["mlx", "mlx-lm", "numpy", "transformers"]
            },
        },
        "method": "serial greedy target; every anchor; disjoint rounds advance accepted+1; EOS stops trajectory; no rollback",
        "nvfp4_actual_export_checks": qualify_nvfp4(draft),
        "prompts": [],
    }
    for category, prompt in PROMPTS[: args.prompt_limit]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        cache = target.make_cache()
        hidden_chunks = []
        for offset in range(0, len(ids), 256):
            hidden, features = target.backbone(
                mx.array([ids[offset : offset + 256]]),
                cache=cache,
                target_layer_ids=taps,
            )
            logits = target.lm_head(hidden[:, -1:])
            sync(cache, logits, features)
            hidden_chunks.append(features)
        initial = mx.concatenate(hidden_chunks, axis=1)
        generated, serial_features, target_ms = [], [], []
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        for _ in range(args.tokens):
            token = mx.argmax(logits[:, -1], axis=-1).item()
            if token in eos:
                break
            generated.append(token)
            start = time.perf_counter()
            hidden, features = target.backbone(
                mx.array([[token]]), cache=cache, target_layer_ids=taps
            )
            logits = target.lm_head(hidden)
            sync(cache, logits, features)
            target_ms.append((time.perf_counter() - start) * 1000)
            serial_features.append(features)
        print(category, "target tokens", len(generated), flush=True)
        context = draft.append_context(initial)
        mx.eval(context.kv, context.anchor)
        # Warm the complete proposal graph without advancing context.
        if len(generated) >= 8:
            for _ in range(3):
                mx.eval(
                    draft.propose(
                        generated[0],
                        context,
                        target.backbone.embeddings,
                        target.lm_head,
                    )
                )
        rows = []
        next_round = 0
        # Keep eight lookahead tokens: seven drafts plus a known bonus on full accept.
        for pos in range(max(0, len(generated) - 8)):
            # Context excludes the anchor: serial_features[pos] is appended AFTER drafting.
            start = time.perf_counter()
            proposed = draft.propose(
                generated[pos], context, target.backbone.embeddings, target.lm_head
            )
            mx.eval(proposed)
            block_ms = (time.perf_counter() - start) * 1000
            proposal = proposed[0].tolist()
            truth = generated[pos + 1 : pos + 8]
            accepted = 0
            for a, b in zip(proposal, truth, strict=True):
                if a != b:
                    break
                accepted += 1
            round_update_ms = None
            if pos == next_round:
                round_taps = mx.concatenate(
                    serial_features[pos : pos + accepted + 1], axis=1
                )
                mx.eval(round_taps)
                start = time.perf_counter()
                updated = draft.append_context(round_taps, context)
                mx.eval(updated.kv, updated.anchor)
                round_update_ms = (time.perf_counter() - start) * 1000
                next_round += accepted + 1
                del updated
            start = time.perf_counter()
            context = draft.append_context(serial_features[pos], context)
            mx.eval(context.kv, context.anchor)
            update_ms = (time.perf_counter() - start) * 1000
            rows.append(
                {
                    "position": pos,
                    "accepted_drafts": accepted,
                    "output_tokens": accepted + 1,
                    "proposal": proposal,
                    "truth": truth,
                    "draft_ms": block_ms,
                    "one_row_context_update_ms": update_ms,
                    "round_context_update_ms": round_update_ms,
                }
            )
        rounds, position = [], 0
        while position < len(rows):
            rounds.append(rows[position])
            position += rows[position]["output_tokens"]
        entry = {
            "category": category,
            "prompt": prompt,
            "input_ids": ids,
            "generated_ids": generated,
            "text": tokenizer.decode(generated),
            "rows": rows,
            "round_positions": [r["position"] for r in rounds],
            "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rounds)
            if rounds
            else None,
            "mean_accepted_drafts": statistics.mean(
                r["accepted_drafts"] for r in rounds
            )
            if rounds
            else None,
            "median_draft_ms": statistics.median(r["draft_ms"] for r in rows)
            if rows
            else None,
            "serial_target_ms": target_ms,
        }
        result["prompts"].append(entry)
        result["summary"] = summarize(result["prompts"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            category,
            "mean output",
            entry["mean_output_tokens"],
            "draft ms",
            entry["median_draft_ms"],
            flush=True,
        )
        del cache, context, initial, hidden_chunks, serial_features
        mx.clear_cache()
    result["verify_curves"] = measure_verify(
        target, tokenizer, args.verify_contexts, draft
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
