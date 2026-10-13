# Harbor handoff: issue #4592 grammar-constrained MTP

- Receiving roles: Vector for backend ownership; Atlas for scope; Echo for issue tracking; Pixel for awareness of user-visible tool reliability.
- Owner for this scoped mitigation: Harbor.
- Branch/worktree: `harbor/4592-grammar-failclosed` / `/private/tmp/harbor-desk-4592-grammar-failclosed`.
- Intention: fail closed from MTP when a grammar logits processor is active, while preserving ordinary constrained decoding.
- Scope: scheduler processor eligibility comments, admission bookkeeping, and focused regression coverage.
- Non-goals: speculative grammar transaction redesign, public API changes, model-specific behavior, or performance claims.
- Planned verification: focused scheduler/MTP tests, Ruff, diff review, and exact-head CI after PR creation.
- Reference check: current speculative constrained-decoding implementations separate tentative validation from committed grammar advancement; the temporary ordinary-decode fallback is safer than changing snapshot boundaries without a failing hermetic transaction test.

## Completion evidence

- `python3.12 -m pytest tests/test_prompt_cache_snapshot.py tests/test_mtp_cli_wiring.py tests/test_reasoning_budget_mtp.py tests/test_repetition_guard.py -q`: 221 passed.
- `python3.12 -m pytest tests/test_mtp_spec_decode.py tests/test_mtp_draft_overlap.py -q`: 200 passed.
- `ruff check rapid_mlx/scheduler.py tests/test_prompt_cache_snapshot.py`: passed.
- `ruff format --check rapid_mlx/scheduler.py tests/test_prompt_cache_snapshot.py`: passed.
- `git diff --check`: passed.
- Remaining work: Vector should add a real-matcher K=2 partial-acceptance regression and diagnose the transaction before restoring grammar MTP eligibility.

This file records the required start/completion FYI because Orca agent messaging is unavailable in this session.
