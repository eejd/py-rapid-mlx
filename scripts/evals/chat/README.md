# Goose Chat QA benchmark subset

`python3 scripts/evals/chat/prepare.py` downloads source data to `~/.cache/rapid-mlx/chat-eval/` and verifies the SHA-256 hashes in `subset.json`. Commit only the IDs and hashes. It selects IDs by SHA-256 of `goose-chat-qa-20261015:<id>`, with Arena restricted to `hard_prompt`, GDPval to three document and two spreadsheet tasks with no reference files, and GAIA level 1 to six supported text/document/spreadsheet attachments plus nine web-only tasks. This avoids choosing questions after viewing model answers. Source records and answers stay out of Git.

| Dataset | Access and terms checked 2026-10-15 | Subset |
| --- | --- | --- |
| [Arena-Hard-Auto v2](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto) | Public, Apache 2.0 | 40 |
| [SimpleQA](https://github.com/openai/simple-evals) | Public CSV and MIT evaluation repository | 30 |
| [Chinese SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA) | Public, CC BY-NC-SA 4.0 on dataset card (the evaluation code has separate MIT terms) | 20 |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | Public, Apache 2.0 | 40 |
| [GAIA 2023 validation level 1](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Gated; terms prohibit republishing validation content in a crawlable location. Accepted local access permits private evaluation. | 15 |
| [GDPval](https://huggingface.co/datasets/openai/gdpval) | Public gold files and rubrics; dataset card has no explicit license tag. Use internally, do not redistribute source files. | 5 |

GAIA: after accepting the dataset terms with your own Hugging Face account, place `metadata.level1.parquet` in the private cache as `gaia-metadata.level1.parquet` and rerun `prepare.py`. The committed manifest contains 15 IDs. The runner fetches six selected attachments only into the private eval workspace. Do not commit the source, answers, or attachments.

Each result record should include `benchmark`, `id`, `path` (`goose` or `direct`), `web` (true/false), `answer`, `first_visible_s`, `total_s`, `tool_calls`, `permission_prompts`, `failure`, and `empty`. Keep raw outputs in an ignored local results directory. Score Arena from blinded, position-swapped comparisons, with ties and unresolved disagreements counted separately. Use the official IFEval checker for its programmatic pass rate. Report SimpleQA labels as correct, incorrect, and not-attempted; human review is required for semantic aliases. GAIA uses its official exact-answer normalization. GDPval deliverables require the public rubrics and gold artifacts for review; existence alone is not a quality score.

A 45-minute run on the 27B is a time budget, not a reason to truncate silently. Report how many records finished under that budget and mark missing scores as unmeasured. Include model revision, MTP setting, host, prompt-token count, and the command used with every summary.

## Runner

Launch an **isolated** 27B MTP engine, the app's `work-inference-proxy.py --goose-chat`, and a Goose `serve` helper with `--with-builtin developer,computercontroller,memory,autovisualiser,tutorial`, using the same provider JSON and environment fields as `WorkAgent.swift`. Use fresh HOME, workspace, and ports; keep Work Full and the installed app untouched. Pass the helper and engine URLs to `run.py`. The runner opens ACP sessions, supplies the bundled `chat-web-mcp.py` for web-on cases, answers permission requests once (eval YOLO), and writes resumable JSONL records. Direct baseline calls the same engine without tools. Put helper and engine secrets in `GOOSE_QA_ACP_SECRET` and `GOOSE_QA_ENGINE_BEARER` environment variables; they are never written to result files.

```sh
python3 scripts/evals/chat/run.py \
  --acp http://127.0.0.1:<own-goose-port> \
  --engine http://127.0.0.1:<own-engine-port> \
  --web-script apps/rapid-mac/Work/chat-web-mcp.py \
  --output scripts/evals/chat/results/run.jsonl \
  --workspaces /tmp/goose-qa-workspaces
python3 scripts/evals/chat/score.py scripts/evals/chat/results/run.jsonl
```

The runner stops at 45 minutes and preserves partial results. It does not turn unscored rows into successes. A second run resumes missing rows. The current revision captures answers and artifacts but does not automate GAIA normalization or GDPval rubric grading; those scores must remain pending until checked.

IFEval strict programmatic scoring uses Google's pinned reference code. In a separate evaluation environment install its requirements (`absl-py`, `langdetect`, `immutabledict`, `nltk`) and run `python3 scripts/evals/chat/check_ifeval.py scripts/evals/chat/results/run.jsonl --output scripts/evals/chat/results/ifeval-labels.json`, then pass that file to `score.py --labels`. The code is cached outside Git.

For human review, run `make_review.py results/run.jsonl --arena-out results/arena-review.json --facts-out results/facts-review.json`. Arena comparisons are shuffled and include both answer positions. The reviewer should use cached gold answers for factuality; result files do not publish benchmark answers. Enter judgments in a separate labels JSON for `score.py`.
