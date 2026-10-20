# Sign-off — Vera (skeptic), oracle-1020-megaprobe-spin-lies

**Answer to the title under the frozen rule (PREREG s6 + D1; D5 void): NOT SETTLED -> UNCONFIRMED. Reproduced.**
The raw rule gives NO for both megaprobes (95% upper bound of m: (c) 0.706, (c2) 0.621, both < 0.80), but (c) failed
the frozen positive control on the untrained model (all lies: c 0.810 < a 0.852), and so did (c2) (0.754 < 0.852).
The rule turns that NO into UNCONFIRMED. With the void D5 (control passes if c OR c2 beats a) the status is the same.

All numbers below come from my own code on the raw files: build's memmaps (`data/cache/*/tok.npy`, `offsets.npy`,
`follow.npy`), the copied 1019 labels (`data/refit_probe/labels_*.jsonl`) and build's saved per-row scores
(`data/fits/*_own.json`). I did not read build's probes/score code.

## Key numbers (spin lies vs truthful answers that mention the fact, mean over the 4 trained models)

| quantity | mine | build |
|---|---|---|
| (c) megaprobe m [95% paired prompt bootstrap] | 0.6757 [0.6415, 0.7059] | 0.676 [0.641, 0.706] |
| gain c - a, g | -0.0566 [-0.0845, -0.0282] | -0.057 [-0.084, -0.028] |
| per model c: lam1_s0 / lam1_s1 / ens_s0 / ens_s1 | 0.661 / 0.702 / 0.697 / 0.643 | same |
| vs ALL truthful: m, g | 0.626 [0.595, 0.655], -0.044 [-0.070, -0.017] | — |
| positive control (base, all lies) c vs a | 0.8101 vs 0.8524 -> FAIL | same |
| (c2) FIBS axial m | 0.5855 [0.5502, 0.6213] | 0.586 [0.550, 0.621] |
| gain c2 - a | -0.1468 [-0.1835, -0.1114] | -0.147 [-0.183, -0.111] |
| per model c2 | 0.622 / 0.607 / 0.557 / 0.556 | — |
| positive control c2 vs a | 0.7539 vs 0.8524 -> FAIL | same |
| (b) layer-average, D7 refit, m | 0.614 (per model 0.589 / 0.618 / 0.656 / 0.595) | 0.614 [0.582, 0.646] |

- Arm (a) exact refit from build's `draws.json` on my own layer-12 means: max |score diff| 4e-5 to 8e-5 (ens_s1 7.6e-4),
  AUROCs equal to 4 decimals on all 5 models (`vera_rebuild.out`).
- Folds/draws leak check (`vera_folds.out`): 0 test rows inside their own fit draw, 0 OTHER rows, draws 400 (200/200;
  base ~180 honest + fill).
- Cache check (`vera_cachecheck.out`): all 9 reads, layer 12/16 answer means vs 1019 `acts_*.npz` cosine >= 0.99996.

## Independent megaprobe (different code, own folds) — `vera_mega.py`, `vera_mega_verdict_fs77.out`

Attention over the 24 layers x answer tokens on per-layer PCA-64 features, 4 heads (~600 params), hyperparameters
fixed on synthetic data before any real score. Own fold map (seed 77), 3 balanced n=400 draws.
- m = 0.703 [0.668, 0.735]; g vs my linear arm = -0.018 [-0.045, +0.007] (a tie, not a loss).
- Per model 0.710 / 0.732 / 0.706 / 0.664. All lies, trained mean 0.776 vs linear 0.804. Base control 0.833 vs 0.838 (near-tie).
- So: no megaprobe we ran gets near 0.80 on spin lies (upper bounds 0.706 and 0.735). "Worse than linear" holds for
  arm (c) only; "no better than linear" holds for both designs.

## Nulls
- Shuffled labels, my linear arms (`vera_lin_fs77_shuffle.out`): mean4 0.50-0.53 on every metric.
- Shuffled labels, my megaprobe (`vera_mega_fs77_shuffle.out`): spin-vs-fact-stated mean4 0.531 [0.496, 0.565], all lies 0.508; my linear arm all lies 0.501 (vera_mega_verdict_fs77_shuffle.out)
- Build's own nulls: a_null 0.45-0.53, c_null 0.42-0.54 (my AUROC on their saved scores).
- Words-only floor (`vera_words.out`, own TF-IDF): mean4 spin vs fact-stated 0.61; (d) build 0.618.
- Fold/draw luck (`vera_lin_fs78.out`): linear arm mean4 spin-vs-fact-stated 0.726 (seed 77) vs 0.707 (seed 78).

## Definition checks
- "Spin lie" = judge DECEPTIVE and string check fact_score >= 0.273 (share of the fact's content words present).
  Median spin lie has 0.33-0.45 of the words; 55-84% have fewer than half. The post must define the class by this
  test (done in blog 55fcd77), not as "states the fact".
- Two negative sets must stay apart: the linear baseline is 0.73 vs fact-mentioning truthful, 0.66 vs all truthful.

## Cautions
1. The status is UNCONFIRMED because of the control, not because of noise: the spin numbers sit far under 0.80.
2. Arm (b) as scored kept 0 layers (fallback to one layer); my sklearn (b) kept 24/24 and scored 0.539. Both weak.
3. c2 at 400 labels is weak even on all lies (trained 0.58-0.66): a 772k-parameter probe trained on 400 rows; FIBS trained on ~153k. Report it as 'did not work at this label budget', not as evidence about the FIBS probe.
4. The one-layer probe reaches 0.801 on lam1_s1: never write 'no probe reached 0.80 on any model'.
5. Held-text clause (iii): see below.

## Pre-registered prediction outcome (s7): FAILED (reproduced)
- (i) gap closure on all lies, arm c: base 0.810 vs trained 0.763 / 0.751 / 0.762 / 0.675 -> gaps 0.047-0.135 > 0.03: FAILS.
- (ii) arm c spin vs ALL truthful < 0.80 on every model (0.599-0.647, base 0.605) and c - a <= 0.08 (all negative): HOLDS.
- (iii) held text within 0.02 of own, arm c (`vera_heldtext.out`): -0.051, -0.055, +0.021, +0.005 -> FAILS as written,
  but every paired 95% CI contains 0 and single-draw AUROCs of c spread up to 0.17 (lam1_s0 0.527-0.693), so the
  failure is inside the megaprobe's fit noise. Arm a held vs own: within 0.006 on all 4 (as in 1019).

## Post numbers checked (blog _notes/megaprobe-spin-lies.md, lines 30-103)
Every number in the main text and foldout matches my own AUROCs (room #3535). Figure data.json: all points and
per-model dots equal mine to < 1e-15 (`vera_figcheck.out`); preview renders readable in light and dark.

## Files
vera_core.py (AUROC, labels), vera_io.py (raw paths), vera_feats.py (own token features from memmaps),
vera_cachecheck.py, vera_folds.py, vera_words.py, vera_lin.py (own a/b/m24/words + nulls), vera_mega.py (own attention
megaprobe + null), vera_verdict.py + vera_mega_verdict.py (s6 rule, paired prompt bootstrap), vera_rebuild.py (build's
scores, exact refit of a, c/c2/b2 verdict), vera_figcheck.py, vera_heldtext.py, novelty.md. `feats/` (1.9 GB derived
arrays) is not committed: rebuild with vera_feats.py from build's memmaps.
