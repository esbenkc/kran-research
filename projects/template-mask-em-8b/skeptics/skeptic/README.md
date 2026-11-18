# Vera (skeptic) receipts — template-mask-em-8b

**NO SIGN-OFF YET.** The 8B runs wait on the laptop queue (labq 3zfskhbdya part 1, fkyekw80z4 part 2).
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

Continuation run (when the laptop jobs are back):
```
uv run python vera_maskrules.py ../../out_p1/gen ; uv run python vera_maskrules.py ../../out_p2/gen
uv run python vera_calls.py answers.jsonl ../../out_p1 ../../out_p2
uv run python vera_check.py answers.jsonl --out vera_check_out.json
```
then compare with analysis.json cell by cell, and check the 8B mask_invariants (check 5b) before trusting any narrow number.
