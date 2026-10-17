# skeptic receipts (lie-type-shift)

Independent rebuild. Did not read or reuse build's code.

- `check.py` — own checker: M, Delta, prompt bootstrap, round/arm permutation nulls. `python3 check.py <labels.jsonl>`
- `out/integrity.py` — label files vs rollouts, duplicates, base identity, judge agreement (kappa).
- `out/null_tests.py` — global label-shuffle null, bootstrap-seed stability, shares, base probe flags by label.
- `myjudge.py` — third judge (claude-haiku-4-5) with my own stricter rubric on base + round 8 (1,000 answers)
  and the 200 DolusChat reference replies. `with-model-keys uv run --no-project --with anthropic python myjudge.py`
- `out/mylabels.jsonl` — its raw labels. `out/rebuild_*.json`, `out/my_check.json`, `out/null_tests.json` — outputs.
- `raw/dolus_eval100.json` — the 100 eval DolusChat rows (public, CC-BY-4.0), matched by
  prompt_id = sha256(system_message + user_query.content)[:12]. Rollouts are read from `../../data/rollouts/`.
- `signoff.md`, `novelty.md`.
