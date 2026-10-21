# Hugo (skeptic2) sign-off — refusal-addback (oracle-1021-refusal)

Model note: the DEV-pick rebuild (pick_rebuild.py, room #4090) ran on openai/gpt-5.5. That session hit its cost cap at
03:38 CEST; every TEST rebuild below ran on Claude Opus with fresh code (hugo_verdict.py, hugo_gap.py, hugo_reader.py).

## What I rebuilt, from raw judged jsonl and acts.npz only

| Quantity | Mine | Build | Code |
|---|---|---|---|
| Base TEST harmful refused | 192/200 = 96.0% [92.3, 98.0] | 96.0% | hugo_verdict.py (base_test) |
| Base TEST harmless refused or garbled | 12/200 = 6.0% [3.5, 10.2]; 188 answered | 6.0% | same |
| DEV pick (D7a) | L*=15, k*=1.5, 18/64 harmful, 3/61 new harmless; L_mid=15; all 27 k*_L match picked.json | same | same (dev_pick) |
| Seed 0 TEST at (15, 1.5) harmful | 75/200 = 37.5% [31.1, 44.4] = 39.1% of base (bar 76.8%) | 37.5% | same (seeds.0) |
| Seed 0 new harmless refusals | 11/188 (bar <= 9) | 11 | same |
| Seed 0 absolute harmless | 12/200 = 6.0% | 6.0% | same |
| Random directions at (15, 1.5) | 15/10/21 of 200, mean 7.7% (bar < 48%) | 7.5/5.0/10.5 | same |
| Own refit direction | 8/200 = 4.0% | 4.0% | same |
| Prereg A / B / C (seed 0) | held / failed / held | same | same (prereg_s0), readout_nulls.json |

## Seeds at the frozen (L*, k*) = (15, 1.5), TEST, judge labels (hugo_verdict.py -> verdict.json)

| Seed | Harmful refused | % of base (bar 80%) | New harmless (bar <= 9/188) | Abs. harmless | Random dirs (mean; bar < 48%) | Refit | Pass |
|---|---|---|---|---|---|---|---|
| 0 | 75/200 = 37.5% [31.1, 44.4] | 39.1% | 11 | 12/200 = 6.0% | 15/10/21 (7.7%) | 8/200 | no (c1, c2) |
| 1 | 64/200 = 32.0% [25.9, 38.8] | 33.3% | 16 | 17/200 = 8.5% | 13/14/20 (7.8%) | 8/200 | no (c1, c2) |
| 2 | 73/200 = 36.5% [30.1, 43.4] | 38.0% | 15 | 18/200 = 9.0% | 15/9/17 (6.8%) | 12/200 | no (c1, c2) |

0 of 3 seeds pass PREREG s8 (with D7/D7a) -> REJECTED: NO to the title. Second scorer (my own substring list) on
harmful at (15, 1.5): 74 / 81 / 55 of 200, also far under the 154/200 bar. Duplicate rows after 1efe1a8: 0.

## Verdict
SIGN-OFF: REJECTED (No to the title). Every seed restores 33-39% of base harmful refusal, against an 80% bar, and adds
11-16 new harmless refusals against a bar of 9.

## Null and robustness tests
- Luck cannot fake condition 1: P(<= observed of 200 | true rate 76.8%) = 2.5e-32 / 4.6e-41 / 7.9e-34 for seeds 0/1/2.
  Condition 2 alone would be fragile (seed 0: P(>= 11 | true 9/188) = 0.29); the No rests on condition 1.
- DEV frontier: all 24 DEV (layer, gain) cells with >= 80% of base DEV harmful refusal cost >= 41/61 new harmless refusals.
- Best TEST layer at its k*_L: L15 (39% of base); next L19 (28.6%).
- Readability null (readout_nulls.py): stripped base-direction AUROC 0.983 at L12 vs 200 random directions P95 0.79 and
  shuffled-label diff-of-means P95 0.92.
- Judge spot check: 125 L15 harmful 'compliance' rows open 'Sure! Here are...' and comply. A naive string list flags 37
  harmless L15 answers that open 'As an AI language model, I can say...' and then comply; the judge is right there.
- Mechanism numbers (hugo_gap.py, hugo_reader.py): DEV L15 gap along r_15 0.789 -> 0.241 |r|, but pooled sd
  0.231 -> 0.107, so separation 3.4 -> 2.2 sd. A cut fixed on DEV (3/64) catches 62.0% of stripped TEST harmful at
  3.0% harmless at L15 (base 93.5% at 5.0%); 64 DEV harmless make that cut noisy (TEST harmless 0.5-11.5% across layers).

## Novelty
See novelty.md. Lee et al. (arXiv 2610.00320, App. H) already wrote a clean refusal direction back over layer bands; the
new part here is one layer, a DEV-frozen gain and a paired harmless budget on an untouched TEST split.

## Definitions
Status cond. 2 is the paired bar of D7/D7a (commits 4be003e / 62fe30e, 02:32 CEST). `git ls-tree` shows no runs/ file
at either commit; the pick commit 5493c54 holds only unsteered TEST rows (400 rows, layer None). Prereg part B stays on
the absolute < 5% bar.
Not stretched: the No comes from condition 1, which no amendment touched.

## Rerun
From the room working dir: `python3 hugo_verdict.py <project>` (status, pick, seeds), `python3 -W ignore hugo_gap.py <project>/runs`,
`python3 -W ignore hugo_reader.py <project>/runs`; readout_nulls.py and base_rates.py / pick_rebuild.py / checker.py are the
gpt-5.5 session's scripts (paths relative to the room dir). acts.npz stays local (50 MB); gap/reader/readout need it.
