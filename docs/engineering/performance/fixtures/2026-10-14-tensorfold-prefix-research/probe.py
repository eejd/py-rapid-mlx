"""Probe token-prefix selection with dummy cache payloads; never load model weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

RUNTIME = "cb2ebf0540f42604e2759b2ddef497861e928248"
SNAPSHOT = "70ae7fac63274ff2eac54152031433374cb80f2f"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RUNTIME:
        parser.error("runtime source must be the qualified revision")
    subprocess.run(
        ["git", "-C", str(runtime), "diff", "--exit-code", "HEAD", "--", "src"],
        check=True,
        capture_output=True,
    )
    repo = next(
        p for p in Path(__file__).resolve().parents if (p / "rapid_mlx").is_dir()
    )
    sys.path[:0] = [str(runtime / "src"), str(repo)]
    from huggingface_hub.constants import HF_HUB_CACHE
    from tensorfold.engine.prefill_plan import PrefillPlan, message_markers
    from tensorfold.families.tokenizer import load_tokenizer
    from tensorfold.server.checkpoints import (
        CheckpointStore,
        choose_checkpoints,
        longest_common_prefix,
    )
    from tensorfold.server.prompt_blocks import PromptBlocks

    from rapid_mlx.speculative.tensorfold_qwen27_server import _prompt_boundaries

    snapshot = (
        Path(HF_HUB_CACHE) / "models--Vontra--Qwen3.8-27B-MLX-4bit/snapshots" / SNAPSHOT
    )
    if not snapshot.is_dir():
        parser.error(
            "qualified tokenizer snapshot must already exist in the standard HF cache"
        )
    tokenizer = load_tokenizer(snapshot)
    app = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_lock=threading.Lock(),
        enable_thinking=False,
        late_system=None,
        effort_for=lambda _: None,
    )
    app.system_prefix_len = lambda messages, tools, tokens, thinking=None: (
        PromptBlocks.system_prefix_len(app, messages, tools, tokens, thinking)
    )
    openers, assistant = message_markers(tokenizer)
    if not openers or not assistant:
        raise RuntimeError("qualified tokenizer failed message-marker detection")
    plan = PrefillPlan(2048, openers, 256, assistant)

    def tokens(messages):
        return tokenizer.encode(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    def store():
        # Payloads have no tensors; entry pressure is logical, never a RAM measurement.
        return CheckpointStore(8, copier=copy.deepcopy)

    def messages(system, user):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def fill(cache, talk, *, nearer=False):
        prompt = tokens(talk)
        chunks = plan.chunks(prompt)
        history, shared = _prompt_boundaries(app, prompt, talk, False)
        if nearer:
            # Counterfactual: add one candidate; rounding still uses existing chunks.
            # A product implementation would reuse the already computed system length.
            system_len = app.system_prefix_len(talk, None, prompt, thinking=False)
            candidate = system_len - plan.min_chunk
            if 512 <= candidate < len(prompt):
                shared = (*shared, candidate)
        hit = cache.match(prompt, usable=lambda n: n in chunks)
        cached = hit[0] if hit else 0
        previous = hit[2] if hit else None
        shared_at = {chunks.floor(n) for n in shared} - {0}
        chosen = choose_checkpoints(history, cached, previous, prompt)
        cuts = sorted(
            n
            for n in {*(chunks.floor(n) for n in chosen), *shared_at}
            if cached < n < len(prompt)
        )
        for n in cuts:
            cache.insert(
                prompt[:n], [{"at": n}], last_prompt=prompt, pinned=n in shared_at
            )
        return {
            "prompt_tokens": len(prompt),
            "history_tokens": history,
            "cached_tokens": cached,
            "suffix_tokens": len(prompt) - cached,
            "chunk_starts": chunks.starts,
            "saved_cuts": cuts,
            "entries": len(cache),
            "evictions": cache.evictions,
        }

    controls = {}
    system = "Use the following reference facts. " + "alpha beta gamma delta. " * 180
    first = messages(system, "Return only the number 4.")
    cache = store()
    controls["repeat"] = [fill(cache, first), fill(cache, first)]
    followup = [
        *first,
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Now return only 5."},
    ]
    controls["followup"] = fill(cache, followup)
    controls["fork_a"] = fill(
        cache,
        [
            *first,
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Branch A: return 5."},
        ],
    )
    controls["fork_b"] = fill(
        cache,
        [
            *first,
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Branch B: return 6."},
        ],
    )
    assert all(
        controls[k]["cached_tokens"] == 911 for k in ("followup", "fork_a", "fork_b")
    )
    assert controls["repeat"][1]["cached_tokens"] == 911
    rows = []
    for repeats in (16, 48, 96, 180, 300, 500, 900, 1700):
        shared_system = (
            "Use the following reference facts. " + "alpha beta gamma delta. " * repeats
        )
        a = messages(shared_system + "\nSession label: red.", "Return only 4.")
        b = messages(shared_system + "\nSession label: blue.", "Return only 5.")
        cache = store()
        cold = fill(cache, a)
        varied = fill(cache, b)
        cache = store()
        candidate_cold = fill(cache, a, nearer=True)
        candidate_warm = fill(cache, b, nearer=True)
        cache = store()
        fill(cache, messages(shared_system, "Session label: red. Return only 4."))
        stable_system = fill(
            cache, messages(shared_system, "Session label: blue. Return only 5.")
        )
        cache = store()
        fill(cache, a)
        edited = fill(
            cache, messages("Changed instructions. " + shared_system, "Return only 5.")
        )
        assert edited["cached_tokens"] == 0
        assert cold["chunk_starts"] == candidate_cold["chunk_starts"]
        assert varied["chunk_starts"] == candidate_warm["chunk_starts"]
        common = longest_common_prefix(tokens(a), tokens(b))
        assert candidate_warm["cached_tokens"] <= common
        rows.append(
            {
                "repetitions": repeats,
                "common_tokens": common,
                "cold": cold,
                "varied_system_tail": varied,
                "metadata_in_user": stable_system,
                "counterfactual_cold": candidate_cold,
                "counterfactual_warm": candidate_warm,
                "early_edit_cached_tokens": edited["cached_tokens"],
            }
        )
    assert rows[3]["common_tokens"] == 913
    assert rows[3]["varied_system_tail"]["cached_tokens"] == 0
    assert [r["counterfactual_warm"]["cached_tokens"] for r in rows[-3:]] == [
        2048,
        4096,
        8192,
    ]
    cache = store()
    for i in range(16):
        fill(cache, messages(f"Conversation {i}: " + system, "Return only 4."))
    pressure = fill(cache, messages("Conversation 0: " + system, "Return only 4."))
    assert pressure["cached_tokens"] == 0 and pressure["evictions"] > 0
    hashes = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "config.json",
        )
        if (snapshot / name).is_file()
    }
    print(
        json.dumps(
            {
                "method": "Real tokenizer and pinned cache selection; dummy payloads; no model inference, TTFT or RAM measurement.",
                "runtime_revision": revision,
                "rapid_revision": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                ).strip(),
                "model_snapshot": SNAPSHOT,
                "python": platform.python_version(),
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("mlx", "mlx-lm", "numpy", "transformers")
                },
                "tokenizer_hashes": hashes,
                "plan": plan.name,
                "controls": controls,
                "rows": rows,
                "pressure_revisit": pressure,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
