# Goose Chat QA benchmark subset

`python3 scripts/evals/chat/prepare.py` downloads source data to `~/.cache/rapid-mlx/chat-eval/` and verifies the SHA-256 hashes in `subset.json`. Commit only the IDs and hashes. It selects IDs by SHA-256 of `goose-chat-qa-20261015:<id>`, with Arena restricted to `hard_prompt` and GDPval to three document and two spreadsheet tasks with no reference files. This avoids choosing questions after viewing model answers. Source records and answers stay out of Git.

| Dataset | Access and terms checked 2026-10-15 | Subset |
| --- | --- | --- |
| [Arena-Hard-Auto v2](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto) | Public, Apache 2.0 | 40 |
| [SimpleQA](https://github.com/openai/simple-evals) | Public CSV and MIT evaluation repository | 30 |
| [Chinese SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA) | Public, CC BY-NC-SA 4.0 on dataset card (the evaluation code has separate MIT terms) | 20 |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | Public, Apache 2.0 | 40 |
| [GAIA 2023 validation level 1](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Gated; terms prohibit republishing validation content in a crawlable location. HTTP 401 without accepted access. | 15 once authorized; currently unavailable |
| [GDPval](https://huggingface.co/datasets/openai/gdpval) | Public gold files and rubrics; dataset card has no explicit license tag. Use internally, do not redistribute source files. | 5 |

GAIA: after accepting the dataset terms with your own Hugging Face account, place `metadata.level1.parquet` in the private cache as `gaia-metadata.level1.parquet` and rerun `prepare.py`. The manifest will then contain the 15 IDs. Do not commit the source, answers, or attachments.

Each result record should include `benchmark`, `id`, `path` (`goose` or `direct`), `web` (true/false), `answer`, `first_visible_s`, `total_s`, `tool_calls`, `permission_prompts`, `failure`, and `empty`. Keep raw outputs in an ignored local results directory. Score Arena from blinded, position-swapped comparisons, with ties and unresolved disagreements counted separately. Use the official IFEval checker for its programmatic pass rate. Report SimpleQA labels as correct, incorrect, and not-attempted; human review is required for semantic aliases. GAIA uses its official exact-answer normalization. GDPval deliverables require the public rubrics and gold artifacts for review; existence alone is not a quality score.

A 45-minute run on the 27B is a time budget, not a reason to truncate silently. Report how many records finished under that budget and mark missing scores as unmeasured. Include model revision, MTP setting, host, prompt-token count, and the command used with every summary.
