# Superseded source CI cancellation

Engine classifier, compute, lane and coverage jobs use `!cancelled()` together
with their existing classification and dependency conditions. This explicit
status function still evaluates routing after a failed dependency, while
allowing an obsolete source run to stop when a newer head cancels it. The
remaining workload jobs retain their full matrices and verification requirements.

The required `tests` and `desktop-tests` aggregates use `always()` to report a
terminal verdict after failures and cancellation. Their normal success steps
run only while not cancelled, and a final `cancelled()` step explicitly fails
the verdict, including cancellation after an earlier success step. Step-level
cleanup and diagnostic uploads retain their existing behaviour.

GitHub re-evaluates job conditions when cancelling a run; jobs with `always()`
may continue. See the [workflow cancellation reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation).
The Desktop workflow already follows this cancellation-aware workload pattern.

When recovering an already running obsolete source:

1. Confirm the PR is open at a different current head and its automatic current
   run is blocked behind the old run in the same workflow/concurrency group.
2. Confirm the old run is this owner's superseded source run. Exclude release,
   nightly and other owners' runs. Record both immutable run/head identities
   and preserve logs.
3. Request ordinary cancellation once. Observe actual terminal state.
4. If cancellation is accepted but job predicates preserve obsolete work, use
   the supported force-cancel endpoint once for that verified obsolete source.
   Do not loop cancellation requests, replay workflows or bypass required gates.
5. Verify the old run is terminal and the existing current run starts. Only the
   current exact head's fresh checks qualify for merge authorization.

Validate deployed cancellation behaviour on a genuine subsequent source update;
local expression/aggregate tests alone do not prove hosted timing improvements.
