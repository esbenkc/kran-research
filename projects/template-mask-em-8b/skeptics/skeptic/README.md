# Vera (skeptic) receipts — template-mask-em-8b

**SIGN-OFF (2026-11-23, room theory-1123-template-mask-em): CONFIRMED on 3 seeds** — see signoff.md (the 2-seed UNCONFIRMED sign-off of 2026-11-21 is 1121/signoff_1121.md).
Everything here was written from PREREG.md (FROZEN RULE, commit de377e046) and the build's OUTPUTS only; I did not
read the build's train/gen/judge/analysis code.

| file | what it does | checked on |
|---|---|---|
| vera_template_ids.py | Qwen3-8B chat-template ids from tokenizer.json + its jinja template | 3 prefix + 9 postfix ids, matches PREREG |
| vera_maskrules.py `<gen root>` | every answer row's `masked`/`masked_gen` vs the frozen arm rules (own tokenizer) | 0.6B smoke: 184 rows, 0 violations |
| vera_calls.py `OUT DIR...` | re-scores every aligned/coherent judge call from the raw top-20 logprobs (Betley formula, mass < 0.25 = NaN), RISKY from the raw reply; denominators from gen rows | judge smoke /tmp/felix-judge: 6 calls, 0 mismatches (tol 1e-3) |
| vera_check.py `answers.jsonl` | rates, cut, habit, gates, status (FROZEN RULE), per-seed permutation null (A vs arm within question), prompt bootstrap (base, A, arm independent), seed-level t(2) interval | known-answer mock: true cut 0.80 -> 0.786, E -0.10; status CONFIRMED (mock_known_answer_out.json) |
| vera_status_power.py / .txt | power of the frozen status rule, 4000 sims | true cut 0.7 -> CONFIRMED 94%; true cut 0.1 -> REJECTED only 54% (t(2)=4.30) |
| novelty.md | closest prior work (Piggyback 2606.06667 Fig 5 + Table 4, TAME 2609.16754, Leong 2502.13946) | |

Continuation run (updated 2026-11-19, room theory-1119-template-mask-em; jobs 57w02sk625 = out_p1 + out_p2,
dz6v4hu8p2 = out_p3; `git pull --rebase` first; run from this folder):
```
P=../..
$P/.venv/bin/python vera_maskrules.py $P/out_p1/gen ; $P/.venv/bin/python vera_maskrules.py $P/out_p2/gen ; $P/.venv/bin/python vera_maskrules.py $P/out_p3/gen
$P/.venv/bin/python 1119/vera_pairing.py $P/out_p1/gen      # and out_p2, out_p3
$P/.venv/bin/python vera_calls.py answers.jsonl $P/out_p1 $P/out_p2 $P/out_p3
uv run --no-project --with scipy --with numpy python vera_check.py answers.jsonl --out vera_check_out.json
```
then compare with analysis.json cell by cell; open out_p*/checks/mask_invariants_seed{0,1}_pad.json and
mask_invariants_seed2.json (pass must be true) before trusting any narrow number; score DEVIATIONS S1 by hand.
The same chain was run on Kasper's mock in 1119/ (README there): it reproduced the planted numbers exactly.

## 1123 (3 seeds; run from this folder after git pull)
```
P=../..
python3 vera_calls.py 1123/answers.jsonl $P/out_p1 $P/out_p2 $P/out_p3          # 20,280 calls, 0 mismatches
uv run --no-project --with numpy --with scipy python vera_check.py 1123/answers.jsonl --out 1123/vera_check_out.json
python3 vera_diff.py 1123/answers.jsonl 1123/vera_check_out.json $P/analysis.json  # 237 comparisons, 0 mismatches
uv run --no-project --with numpy python 1123/vera_1123.py 1123/answers.jsonl      # exact means, seed agreement, rule null, LOQO
python3 1123/vera_ties.py $P/out_p1 $P/out_p2 $P/out_p3                          # judge near-tie fragility of habit
```
