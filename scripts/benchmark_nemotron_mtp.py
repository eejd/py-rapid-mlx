# SPDX-License-Identifier: Apache-2.0
"""Locked, lease-owned Super native MTP qualification (no model downloads).

Run under lockf -k /var/tmp/rapid-mlx-large-model.lock after claiming a lease.
Results include token IDs and per-depth acceptance; prefill is excluded from
steady decode throughput. Each condition stops at EOS or the same token budget.
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step
from rapid_mlx.spec_decode.mtp.nemotron_h_inject import inject_mtp_support
from rapid_mlx.utils.tokenizer import load_model_with_fallback

PROMPTS = {
    "chat": "Explain how to choose a good name for a variable. Give practical examples.",
    "code": "Write a Python LRU cache with get and put methods and explain its complexity.",
    "math": "Find the sum of all integers from 1 to 1000. Show the derivation and verify it.",
    "rewrite": "Copy the following paragraph exactly, then rewrite it in the past tense: "
    "The engineer opens the window, checks the weather, and walks to the station. "
    "She buys a ticket, boards the train, and reads a book until she arrives.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    model, tokenizer = load_model_with_fallback(args.model)
    mx.eval(model.parameters())
    assert inject_mtp_support(model, args.sidecar)
    stop_tokens = set(getattr(tokenizer, "eos_token_ids", set()))
    result = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "platform": platform.platform(),
        "mlx": mx.__version__,
        "model": args.model,
        "sidecar": args.sidecar,
        "rows": [],
    }
    filler = tokenizer.encode(
        "Reference notes: clear names describe the purpose of data. "
        "Examples include a train timetable, a cache, and an arithmetic series.\n"
    )
    for context in args.contexts:
        for category, instruction in PROMPTS.items():
            # Fixed token count, with the task at the end of a valid chat turn.
            task = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            padding = (filler * (context // len(filler) + 1))[: context - len(task)]
            prompt = mx.array(padding + task, mx.uint32)
            for rep in range(args.repeats):
                # Alternate ordinary and speculative conditions to limit drift.
                for depth in [0, 1, 2] if rep % 2 == 0 else [2, 1, 0]:
                    mx.random.seed(42)
                    counter = MTPAcceptCounter()
                    if depth:
                        gen = mtp_generate_step(
                            prompt,
                            model,
                            max_tokens=args.tokens,
                            prefill_step_size=256,
                            temp=0,
                            max_k=depth,
                            disable_auto_k=True,
                            accept_counter=counter,
                            prompt_lookup_enabled=False,
                            stop_tokens=stop_tokens,
                        )
                    else:
                        gen = generate_step(
                            prompt,
                            model,
                            max_tokens=args.tokens,
                            prefill_step_size=256,
                            sampler=make_sampler(temp=0),
                        )
                    started = time.perf_counter()
                    first = None
                    tokens = []
                    try:
                        for item in gen:
                            token = int(item[0])
                            if first is None:
                                first = time.perf_counter()
                            tokens.append(token)
                            if token in stop_tokens:
                                break
                    finally:
                        gen.close()
                    mx.synchronize()
                    end = time.perf_counter()
                    row = {
                        "context": context,
                        "prompt_tokens": prompt.tolist(),
                        "category": category,
                        "repeat": rep,
                        "k": depth,
                        "tokens": tokens,
                        "ttft_s": first - started,
                        "decode_s": end - first,
                        "tok_s": (len(tokens) - 1) / (end - first),
                        "acceptance": asdict(counter.snapshot()),
                        "peak_bytes": mx.get_peak_memory(),
                    }
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(
                        json.dumps(
                            {key: val for key, val in row.items() if key != "tokens"}
                        ),
                        flush=True,
                    )
                    mx.clear_cache()
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
