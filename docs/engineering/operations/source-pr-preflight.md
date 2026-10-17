# Retired source PR CPU preflight

Ordinary PRs now run the ordinary path-aware gate, including applicable Apple
Silicon unit tests and changed-line coverage. `RAPID_MLX_SOURCE_PREFLIGHT` and
source canary variables no longer affect ordinary PR classification or opt PRs
into reduced validation. See [merge operations](path-aware-merge-queue.md) for
current policy.

The remaining content describes the retired reduced route. Its implementation
is kept in the tree until the queue removal follow-up; this is historical
context, not an activation or rollback procedure.

## Retired gate placement

Preflight ran all three ordinary/headless unit shards on Python 3.11, merged
all three Linux coverage artifacts, and retained lint, type, engine contracts and
the MLX dependency guard. It did not select only a mapped subset of unit tests.

Python version compatibility, Apple execution, model smoke contracts and the
100% changed-line coverage requirement were enforced on the actual combined
merge candidate. Source preflight therefore deferred source Apple and changed-line
jobs. Passing the source `tests` aggregate admitted a PR to normal queue validation;
it did not qualify a full evidence record or permit merging a failed candidate.
The full evidence validator still required nine CPU and five model identities,
Apple success and candidate coverage. Candidate, main and release policy were
unchanged by the source-only route.

Diffs confined to `rapid_mlx/` and ordinary top-level `tests/test_*.py`
files qualified, including their accompanying documentation. Existing documentation
paths (`docs/`, README, contributor/agent/security guides and license) did not
broaden source CPU preflight into the full source route. Documentation alone
retained the documentation lane and could not select engine preflight. Source
controllers, CI tests, collection support, dependencies, unknown paths and
cross-product changes retained the full source route. Invalid or traversal paths
failed closed even when they resembled documentation. Runtime hot paths could
receive this CPU prefilter because their combined candidate still required full
execution before merge. Forks, promoted `train/*` or managed queue branches,
merge groups and main retained full validation. The mapped source canary took
precedence when its own exact-base qualification succeeded; the two reduced
source routes were mutually exclusive.

## Retired verification and measurement

The retired route required independently reviewing the policy head, validating
its normal source checks, and verifying actual combined-candidate job identities
and coverage. Source classifier output selected Python 3.11 shards 1–3; full
candidate and changed-line coverage remained required. The combined candidate
still had to pass the nine CPU and five model jobs and required Apple/coverage
gates before merge.

Source creation, execution, queue admission, candidate creation and actual merge
were measured separately. The route removed duplicate source GPU/compatibility
work; it did not establish a per-PR time guarantee or remove serial queue waiting.
