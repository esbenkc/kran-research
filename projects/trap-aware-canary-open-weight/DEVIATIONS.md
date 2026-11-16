# Deviations and v1.1 facts — trap-aware-canary-open-weight

PREREG.md is recorded at commit 628f3f93 (GitHub push 2026-11-16T07:03:27Z, header 64bb1779). Nothing in its
FROZEN RULE block has changed. Entries below are either v1.1 facts allowed by PREREG.md (code sha, job ids, smoke
numbers, seed formula) or harness changes made before any counted run.

## v1.1 facts (Bruno, build)

- **Per-step seed, as coded** (`web/run.py`, `web/agent.py`, `web/mlxcall.py`): seed group s = i mod 3; run seed
  = 1000 * s + i; each agent step calls `mx.random.seed(run_seed * 1000 + step)` before sampling. The seed is a function
  of (s, i, step) only, as PREREG s1 says, so the warned and the unwarned run with the same index share a seed stream
  (common random numbers). The prompts differ in the sentence, so the sampled tokens differ from the first step.
- **gpt-oss prompt**: the chat template renders our system prompt (with the sentence) as the harmony DEVELOPER message
  under `# Instructions`; the harmony system message holds the model identity, `Knowledge cutoff: 2024-06`,
  `Current date: 2026-11-16` (pinned by the harness; the template prints today's date) and `Reasoning: medium`.
  Every step-0 record keeps the first 1,500 rendered characters (`replies[0].meta.rendered_head`).
- **Harness checks** (`web/test_harness.py`, no model): ALL OK, including the minimal pair (the absent page equals the
  present page minus ` aria-hidden="true" style="display:none"`), the priority list counts (P1 2 x 60, P2 2 x 40,
  P3 2 x 20, P4 2 x 40), the seed balance (20/20/20, every seed group covers all 5 templates), the reply splitter
  (Qwen3 and gpt-oss reasoning never reach the parser; a harmony tool call is a format failure with its own flag).
- **Mini smoke** (smoke runs, never counted; `web/out/smoke-*`):
  - Qwen3-0.6B (4-bit), 12 runs (2 run indices x every cell): 23-55 s per run, RSS 0.40-0.57 GB, MLX peak 1.47 GB;
    the agent re-fetched the home page for all 8 steps in every run.
  - Qwen3-1.7B (4-bit), 12 runs (strict parser) and 16 runs (`smoke2`, last-`</think>` split): 10-49 s per run,
    RSS 0.55-1.28 GB, MLX peak 1.85 GB; 31 of 57 steps failed the parent's strict parser in the first smoke (see D1).
- **Laptop jobs**: code sha and labq ids are posted in the room at submission and added here.

## D1 — lenient first-object parse and the last `</think>` (pre-data harness change; Felix #22470)

Before any counted run. Not in the FROZEN RULE block. Backed by Vera #22469, Tove #22472, Iris #22471.

- **What changed.** (a) Qwen3: the final text is the text after the LAST `</think>` (the Qwen3 model-card parser),
  not after the first. (b) All agents: when the parent's strict parser (`json.loads` of the whole final text) fails,
  the harness takes the FIRST complete JSON object in the final text (`json.JSONDecoder.raw_decode` at each `{`) that
  passes the parent's tool and args checks (Kasper #22473: an example object quoted in prose is skipped). Only the
  final text is parsed, never the reasoning.
- **Why.** In the Qwen3-1.7B smoke, 31 of 57 steps failed the strict parser. Most were a valid action followed by a
  stray `}` or a code fence. The error message entered the transcript, and the model then copied its broken reply until
  the step cap: a loop that ends the run without a follow. The lenient parse rescues 18 of those 31 steps. A
  formatting slip is not a choice about the link, and strict parsing would turn it into a fake non-follow that differs
  by model (the capability confound named in Phase 0).
- **Rows kept.** Each step records `parse` = strict | lenient | fail. score.py reports `strict_format_fail` per run;
  analysis.py reports `fit_e` on runs with 0 strict failures per cell as a sensitivity row (`fit_strict_runs`).
  The fit rate the strict parser would have produced cannot be rebuilt offline: after a strict failure the parent's
  error message enters the transcript and the run diverges, so the sensitivity row is the only strict reading.
- **Parent comparison.** The parent's 3 API agents ran under the strict parser. Parse failures touched 2 of their
  counted web runs (parent `web/results/summary_counted.json`), so their rows in the figure are strict-parser rows
  and the new rows are lenient-parser rows; the difference cannot move a parent row by more than 2 runs.
