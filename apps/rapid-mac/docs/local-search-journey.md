# Local Search desktop journey and handoff

## Grouped All results completion

The All tab requests `group_limit: 8` and consumes optional ordered groups.
Documents show up to five existing rows; Images and Videos show up to eight
existing media cells in one horizontal row. Headers announce the kind and
admitted count to VoiceOver. Show all switches to the corresponding full tab.
Older responses without groups retain the flat view and per-kind requests
retain their existing limit of 50.

Up/Down moves through document rows and crosses section boundaries. Left/Right
stays within the selected media row. Crossing between media sections preserves
the column where possible; entering documents selects the nearest boundary row.
Return, Space, Cmd-Return and Cmd-C operate on the displayed selection.

New accessibility identifiers are `LocalSearch.Section.{text,image,video}`,
with `.Header` and `.ShowAll` suffixes. Result identifiers remain
`LocalSearch.Result.<displayed index>` across sections.

Completion verification: all 59 LocalSearch Swift tests pass, including decoding,
legacy fallback, preview order/counts/limits, selection boundaries, and native
headers, video time labels and Show all. All 182 engine tests pass. The saved
80-query and 48-query replays confirm that grouping preserves every existing
flat result. Owner handoff: Pixel owns the section presentation and keyboard
behavior; Vector owns group ranking and counts. Atlas, Harbor and Echo FYI:
the task remains on `local-search-mvp` in this worktree, with no commit, push,
PR, release or model download. Orca messaging is unavailable in this session.

Owner: Pixel. Host: Studio Mac. Branch: local-search-mvp. Worktree: 嵌入式.
Scope: apps/rapid-mac only; consume the HTTP contract without changing it.
No commits, pushes, pull requests, dependencies, engine implementation changes,
or model downloads are part of this work.

## Walkthrough findings and changes

- Enabling a spawn-time flag previously exposed an unusable destination. A live
  child now gets one clearly labelled restart confirmation; Cancel reverts the
  opt-in. The restart uses the same stop/ensure-serving path as other spawn
  settings and warns that other resident models unload.
- Search previously sent a stopped-engine user to Chat. It now starts the
  current model, falling back to the last model/default setup choice, through
  the same readiness start function as Chat. Progress, startup rejection,
  retry, and missing-engine recovery stay on the Search surface.
- First-run choices were hidden inside a menu. Checkable Desktop/Documents/
  Downloads rows, Whole Mac and a native multi-folder picker are directly
  available. The privacy statement and optional search-model download card
  appear in the same scrollable surface, avoiding clipping at larger text sizes.
- Whole Mac's time expectation was hidden in a tooltip. It is now visible,
  explicitly saying that search remains usable during indexing.
- Indexing progress disappeared when folders collapsed. A compact persistent
  line now shows processed files/current folder, uses an indeterminate indicator
  for zero totals, and keeps its space after completion. Results have a reserved
  hint line explaining incomplete coverage during indexing.
- Empty responses during indexing previously looked like a blocking index
  state. They now describe searched folders/kinds and choose a clear-filter,
  continue-indexing, or add-folder next step.
- Typing cleared results/selection. Debounce now retains both, preserves a
  surviving selection, and rejects stale responses. Discovery-only status
  changes no longer rerun searches. Index-content changes still refresh results.
- Rows showed absolute user-home paths and fragmented accessibility content.
  They now show the containing folder relative to the indexed root, full path
  in the tooltip, filename/extension, native type icon or thumbnail, highlighted
  snippet, page/line, and modified date. VoiceOver gets one sentence per document.
- Copy Path was missing. It joins Open, Quick Look and Reveal in Finder in the
  context menu, with Cmd-C when results have keyboard focus. Arrow keys transfer
  focus from the query field; Return opens, Space toggles Quick Look, Cmd-Return
  reveals. Cmd-Shift-F also refocuses an already-visible Search view via both
  sidebar layouts. Native file opening reports failure instead of failing silently.
- Folder summaries exposed internal chunk counts and unexplained "skipped"
  counts. They now say "N files indexed · M not indexed". Missing/error/waiting
  states each have a plain-language explanation and recovery action. Removal is
  pending until confirmed and explicitly leaves source files untouched.
- Document support was left implicit. The first-run surface now explains
  searchable PDF text and .docx, plus the need for readable text in scanned PDFs
  and conversion of older .doc files. Extraction/OCR implementation stays with
  Vector; the desktop does not invent document-processing capabilities.
- Direct-choice accessibility identifiers initially propagated over their
  children. Explicit containment keeps every checkbox/action discoverable.
  Semantic fonts and the existing text-size clamp are reused; there are no
  introduced motion animations or animated scrolling.

## Contract handoff to Vector

The currently observed status payload has `files_skipped` but no reason breakdown.
The desktop optionally decodes `folders[].skip_reasons: {string: integer}`;
request human-readable reason labels with file counts, e.g.
`{"Permission denied": 3, "No readable text": 2}`. Missing/null means count only.
This addition is optional and is not required for the rest of the journey.

The system default file-opening API has no general PDF page-selection parameter.
PDFs therefore open normally in the user's configured handler; the matched page
remains visible in the result. No undocumented commands or file-URL fragments
are used.

## Verification and team awareness

Run `swift test --disable-sandbox --package-path apps/rapid-mac --no-parallel --filter 'LocalSearch|SpawnArgumentsTests|ServerRuntimeCapabilitiesTests|Sidebar'`.
Coverage includes stopped/starting/crashed/retry, loading/stale status, unsupported,
no folders/model missing, download completion/reindex/retry, indexing with unknown
and known totals, results/no results, filters, debounce/cancellation/selection,
folder recovery/counts/optional reasons, removal confirmation and accessibility.

Team start/completion FYI delivery is unavailable in this session: there is no
bound orchestration Run or current role recipients. This document carries the
intention, scope, ownership, evidence and Vector handoff for later delivery to
Atlas, Vector, Harbor and Echo. Review/production rollout remains outside scope.

Final required-filter run: 132 tests passed in 16 suites; diff whitespace checks
passed. Additional readiness/localization checks: 12 passed; the existing
experimental-surface native string extractor timed out in `xcstringstool` after
30 seconds (sources outside Local Search). Catalog JSON, translations, format
arguments and compiled zh-Hans lookup checks passed.
