# skeptic sign-off — "Would a Random Network Pass the 'One Main Job' Neuron Audit?"

Verdict: **SIGN-OFF on the numbers. The pre-registered predictions H1-H4 all FAILED. The post must say so.**

All numbers below are recomputed by my own code (`rebuild_judgefree.py`, `rebuild_rates.py`,
`null_label_shuffle.py`) from `data/contexts.jsonl` and the RAW judge text in `data/judgments.jsonl`
(`label_raw`, `det_raw`), not from `results.json`. I did not read the build/research code.

## 1. Data hygiene (judge-free, 450 units)
- 0 violations: 20/20/20 example/heldout/random counts; at most 1 document in each unit's top-40;
  every excerpt at position >= 4; top-20 sorted; held-out below rank 20; random docs disjoint from top docs.
- Sink leak: A and C top-20 at positions 4-7 = 1.7-1.8% (uniform 1.6%). B has an early-position excess
  (6.8% at pos < 8, 28% at pos < 32 vs 11% uniform; 3/150 units with >= 5/20 at pos < 8). A random net
  peaks near sequence start. This is a property of the random arm, not a leak; it cannot raise B's rate (B is 0%).
- Lexical purity >= 0.75 (exact token string, as the prereg words it): A 3.3%, B 0%, C 0%.
  Research reports A 4.7%: that matches my **normalized** (strip + lowercase) purity. Say which one the post uses.

## 2. Rates (unit score = mean of 2 judges; 95% bootstrap CI over units; one-sided permutation p)
| arm | one main job (strict) | non-lexical | + related offshoots | fit-only (no gate) | AND of judges |
|---|---|---|---|---|---|
| A trained neurons | 14.3% [10.0, 19.0] | 1.7% [0.3, 3.7] | 27.0% | 15.7% | 6.0% |
| B random-init net | 0.0% [0, 0] | 0.0% | 0.0% | 1.3% | 0.0% |
| C random directions | 1.7% [0.3, 3.3] | 0.3% | 4.7% | 2.3% | 0.0% |

- Gaps (strict, any type): A-B = 14.3 pp [10.0, 19.0], p = 1e-4; A-C = 12.7 pp [8.0, 17.3], p = 1e-4.
- Offshoot variant: A-B = 27.0 pp [21.0, 33.3]; A-C = 22.3 pp [15.7, 29.0].
- A per layer (strict): L4 23%, L14 10%, L24 10%. Per judge: haiku 18.7%, gpt-4.1-mini 10.0%.
- Kappa (one-job verdict) 0.344 over all 450 units, raw agreement 93%. Parser first-vs-last JSON: identical rates.
- These match research's posted numbers exactly (#906).

Pre-registered hypotheses: H1 nonlex A-B = 1.7 pp (needed >= 25) FAILS. H2 nonlex A-C = 1.3 pp (needed >= 15)
FAILS. H3 B any-type = 0% (needed >= 40%) FAILS. H4 A any-type = 14.3% (needed 50-90%) FAILS.

## 3. Null test: label shuffle (pre-registered), my own caller at temperature 0
Each unit's 20 top excerpts re-judged against another unit's label (same arm, same layer, seeded derangement),
plus the frozen detection prompt with that mismatched label. Arms A and C, 25 units per layer, both judges
(285 usable unit-judge pairs). B skipped: B has no one-job verdicts, so leniency cannot inflate it.

| arm | fit >= 15, own label | fit >= 15, shuffled | det >= 4/5, own | det >= 4/5, shuffled | one-job, shuffled |
|---|---|---|---|---|---|
| A | 7.7% | 0.7% | 57.0% | 21.8% | 0.7% |
| C | 0.7% | 0.7% | 29.4% | 20.3% | 0.7% |

- Pre-registered alarm (fit >= 15 for > 10% of mismatches): **not triggered** (0.7%). The rubric is not lenient.
- The detection gate passes a WRONG label 21% of the time, 2x blind chance (10.3%). Alone it is weak; combined
  with fit >= 15 the false-pass rate is 0.7%.
- My strict independent re-count with the unit's OWN label gives fit >= 15 for only 7.7% of A pairs (judges'
  self-reported counts: 11-20%). So if anything the judges' self-counts are generous; the low A rate is not an artefact of a harsh judge.

## 4. Definition-stretch check
- I read every passing label. B: 4 verdicts reached fit >= 15 with vacuous labels ("tokens with guillemets",
  "words or punctuation marks"); the detection gate rejected all 4 (hits 1-2/5). The gate does real work on B.
- A: 3 of 43 passing judge verdicts are broad ("common English nouns", "proper nouns", a list of function words);
  C: 2 of 5 ("a common English word", "substring within a word"). Removing them: A 13.3%, C 1.0%. No conclusion changes.
- Judge TYPE labels are unreliable (gpt-4.1-mini calls most units LEXICAL; judge "LEXICAL" mostly means a
  token class, e.g. "digit 0", "function words"). The non-lexical contrast (H1/H2) is therefore mainly a measure of
  judge habit. Do NOT rescue H1 by switching to any-type or to the offshoot variant after the fact: the post must
  report H1 as failed and present any-type / offshoot gaps as exploratory.

## 5. What the post may claim (and may not)
- MAY: under this blind protocol, 0/150 random-network neurons pass; 14.3% of trained Qwen3-0.6B neurons pass
  (27% counting related offshoots); random directions 1.7%. The audit separates trained neurons from a random
  network and from random directions (exploratory gaps, p < 1e-3).
- MAY NOT: that this replicates or refutes the essay's ~90%. Protocol is a lower bound (0.6B, positive pole,
  top-20 64-token windows, no steering, no circuits). Must say n = 1 random init and 1 draw of directions.
- Status: all four pre-registered predictions failed -> I recommend `lab_status: "NULL"` with the null stated as
  "the predicted 25 pp non-lexical gap did not appear; random nets get 0%, trained neurons only 14%".

API spend (skeptic): $2.29 (includes ~$0.4 wasted on a too-short detection token budget, re-run).

## Reproduce (from this folder)
- `python3 rebuild_judgefree.py ../../data/contexts.jsonl` -> out/rebuild_judgefree.txt
- `python3 rebuild_rates.py ../../data/judgments.jsonl last` (or `first`) -> out/rebuild_rates_*.txt
- `with-model-keys uv run --no-project --with openai --with anthropic python null_label_shuffle.py --contexts ../../data/contexts.jsonl --labels labels_AC.jsonl --detect ../../code/judge/detect.md --n-per-cell 25`
  (all my judge outputs are in out/raw_calls.jsonl; out/extra_checks.txt has the gap CIs, variants and every passing label)

## Addendum (after draft v3)
- Fit-cut sweep (my code, same verdict rule, detection gate kept): one-job % at cut 8/10/12/15/18 =
  A 43.7/37.7/30.7/14.3/5.0, B 5.7/1.7/0/0/0, C 18.7/13.3/9.3/1.7/1.0. Matches skeptic3. Ordering A > C > B at
  every cut; A's level is set by the cut, so the post must not put 14% against the essay's 90% like-for-like.
- Even over all job types the pre-registered gaps miss: A-B 14.3 pp (< 25), A-C 12.7 pp (< 15). H1/H2 did not fail
  only because of the lexical split.
- The 'first item of a pair' unit (7fb2f9cd3e): by my count that reading fits ~10 of 20, below 15. Near miss, not a hidden pass.
