# Merge on green operations

Owner: release engineering.

Provision `RELEASE_PAT` with merge and revert branch/PR write access before
enabling the merger. Missing credentials fail closed. Set
`MERGE_ON_GREEN_ENABLED=false` to disable new automated merges.

Apply exactly one ready label after review and green PR-head checks:
`merge-ready` when `merge-lane-no-mac` succeeds and `merge-lane-mac` is skipped,
or `merge-ready-mac` when `merge-lane-mac` succeeds. The `merge-ready-head`
status binds authorization to the exact head. After another push, remove and
reapply the appropriate label. Remove the ready label to withdraw authorization.

With `MERGE_ON_GREEN_ENABLED=true`, the merger squash-merges authorized green
heads directly, one at a time, without requiring a rebase. Inspect its step
summary for decisions. Version-bump PRs require human merges. Main-guard revert
PRs require human review and explicit merge authorization.

Ordinary PRs run path-aware Linux, Apple Silicon and 100% changed-line coverage
checks. Same-repository `train/*` heads retain full validation in their selected
lanes. Every main push, nightly run and manual dispatch runs the full engine and
Desktop gates, including L1 smoke and GUI golden flows. Release gates still
validate the exact release SHA.

Set `RELEASE_FREEZE=true` before release-sensitive manual work and remove it
only after validation and recovery. Active or undetermined release runs also
freeze merges automatically; API failures fail closed. Neither switch cancels
an already submitted merge request.

For a red main push, inspect the main guard's rerun and recovery issue or revert
PR. Scheduled failures use the nightly tracking issue. Do not authorize recovery
until the relevant evidence is green. See the
[decision and rollback procedure](../decisions/2026-10-17-merge-on-green.md).
