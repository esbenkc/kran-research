# Vera (skeptic) sign-off: theory-1104-transcoder-neuron

Task reviewed: Bruno's "engine + READ/WRITE runs" (kran-research 2b6161f0, projects/transcoder-neuron, PREREG v2.6).
Verdict: **SIGN-OFF: UNCONFIRMED (reproduced)**. The frozen status rule gives UNCONFIRMED. All four layers are NEITHER.
I rebuilt every headline number with my own code. I did not read or run any teammate script.

## 1. Physics rebuild from the model (own engine)

`vera_engine.py` is my own GPT-2 forward pass. Its logits match HF to max |d| 1.8e-4. The real gradient matches a finite difference to 1%.
The circuit-tracer linearisation has frozen attention patterns and LN scales, plus downstream TC with a frozen error term.
`vera_rebuild.py` re-derived 400 pushes: 25 per layer x group, drawn at random from Bruno's CSV.
For each push I recomputed pre, du, MLP and TC response, g_real, g_graph and the end-to-end logit change.

| column | median rel. diff | p95 rel. diff | corr |
|---|---|---|---|
| pre-activation | 2.5e-6 | 4.0e-5 | 1.0000 |
| meas_local_r (PRIMARY measured) | 6.2e-4 | 0.012 | 0.99999988 |
| total_r (PRIMARY predicted) | 4.7e-4 | 0.0098 | 0.9999982 |
| meas_local_g / total_g (graph g) | 3.9e-4 / 2.7e-4 | 0.005 / 0.004 | 0.99999997 |
| e2e_real (DEFINED gate) | 7.7e-4 | 0.031 | 0.99999999 |
| tr2, r2 (Prediction C) | 1.3e-5, 1.0e-5 | 2e-4 | 1.0000 |

Source: `vera_rebuild.json`, `vera_rebuild_rows.csv`. I re-tokenised 50 windows from pile-10k and all 50 match `windows.npy`.

## 2. Statistics rebuild from raw CSV (`vera_stats.py` -> `vera_stats.json`)

PRIMARY R^2 about y = x, pooled over 3 seeds. CIs come from a bootstrap over units.

| layer | NEURON | REST | GAP | ORACLE (real neuron) | DEFINED | seed verdicts | verdict |
|---|---|---|---|---|---|---|---|
| 8 | 0.659 [0.62, 0.75] | -1.126 [-2.77, 0.09] | 0.658 | 0.174 | 0.818 | N/N/N | NEITHER |
| 9 | 0.943 [0.81, 0.97] | 0.591 [0.30, 0.76] | 0.855 | 0.787 | 0.843 | Y/NO/N | NEITHER |
| 10 | 0.447 [0.23, 0.62] | -0.148 [-0.55, 0.17] | 0.152 | 0.260 | 0.787 | N/N/NO | NEITHER |
| 11 | 0.631 [0.36, 0.76] | -0.552 [-1.60, -0.04] | 0.644 | 0.408 | 0.882 | Y/NO/N | NEITHER |

All of these match Bruno's status.json to 3 decimals. Status = UNCONFIRMED. Prediction A failed, because NEURON >= 0.7 only at L9.
- Class-label shuffle null for the gap NEURON - REST (500 permutations of unit labels within each layer): two-sided p = 0.20 / 0.056 / 0.034 / 0.032. Bruno's p values are 0.17 / 0.048 / 0.041 / 0.019.
- Prediction B: my pooled partial Spearman is -0.0947 (Bruno: -0.094). The rule gives FAILS (> -0.1). **Fragile:** the bootstrap 95% CI over features is [-0.128, -0.060], and P(<= -0.1) = 0.38 (`vera_extra.py`). The verdict FAILS vs UNSETTLED depends on the resample. HOLDS (<= -0.2) never happens in 1,000 resamples. The shuffle null 95% band is [-0.033, +0.031], so the law is real but weak. Inside the gap zone alone, the partial Spearman is -0.026, so there is no smooth trend from 0.5 to 0.9.
- Prediction C: 4 of 12 cells fall within 0.15 -> FAILS. This matches Bruno's result.
- Scout v1 (WRITE arm): it fails both with the graph g and with g_real. With g_real, NEURON = 0.90/0.95/0.92/0.995 and REST = 0.80/0.78/0.90/0.97, so the arm cannot tell the classes apart, as predicted. With the graph g, NEURON = -2.6/-0.8/0.26/0.97 and REST = -37/-8.7/-4.7/0.71.

## 3. Definitions: were any stretched?

- Cosines: Bruno computes them on CENTRED e_f and centred folded W_in columns. He announced this before the data (#14399 b). With that rule my numbers match to 1e-6 (`vera_cosdef.json`). The PREREG text says "max_n |cos(e_f, g2 W_in[:, n])|" and does not mention centring. Under the uncentred rule, 0.1/0.3/1.2/1.1% of the sampled units (L8-L11) would change NEURON membership. That is too small to move any verdict, but the post should name the rule ("cosines of centred vectors, because the transcoder input has zero mean").
- The ORACLE row is not a ceiling. Real neurons score BELOW neuron copies at L8, L10 and L11 (0.17/0.26/0.41 vs 0.66/0.45/0.63). The post must call the row "real neurons as features", not a ceiling.
- Dose confound check: REST pushes are about 2x larger (|du|/|x_n| median 0.017 vs 0.007 at L8). At NEURON's dose distribution, REST R^2 is -0.20 / 0.36 / -0.26 / -0.57 (median of 200 resamples). So the gap is not a dose artefact.
- Tail: dropping the 2 worst units of 300 moves L8 NEURON 0.659 -> 0.708 and REST -1.13 -> -0.07. This agrees with Kasper #14442. The class R^2 values rest on a few features; the post must say so.

## 4. Null tests run

- Class-label shuffle (section 2).
- Random-push reference at matched |du|: 0.25/0.32/0.12/0.25 (NEURON rows). It is a reference line, not a floor (#14410).
- Shuffle null for the law (permute joint cosine within each layer, 300 times): [-0.033, 0.031].
- Pre-data screens (#14387, #14410) found the f-only metric fails on real neurons (0.33/0.07/-0.16/-0.34). They also found pooled R^2 is unstable across disjoint texts. Both screens were posted before the freeze and are disclosed in PREREG.

## Proof-matrix rows (skeptic)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0-2 | rebuild, 400 pushes recomputed from model | none | PRIMARY columns match, median rel 6e-4 | skeptics/skeptic/vera_rebuild.json |
| headline | 0-2 | stats rebuild L8-11 | none | 4x NEITHER, UNCONFIRMED | skeptics/skeptic/vera_stats.json |
| gap | 0-2 | L8-11 | class-label shuffle | p 0.20/0.056/0.034/0.032 | skeptics/skeptic/vera_stats.json |
| law (B) | 0-2 | pooled | bootstrap + joint-shuffle | -0.095 [-0.128,-0.060], null +-0.03 | skeptics/skeptic/vera_extra.json |
| gap | 0-2 | dose-matched | dose matching | REST at NEURON dose -0.20/0.36/-0.26/-0.57 | skeptics/skeptic/vera_extra.json |
| class rule | - | uncentred cosine | definition | 0.1-1.2% units flip | skeptics/skeptic/vera_cosdef.json |

## How to rerun

`uv run python vera_stats.py` (4 s), `vera_extra.py` (3 s, needs vera_cosdef.json), `vera_cosdef.py` (3 s), `vera_rebuild.py 25` (71 s, 2.7 GB peak).
The scripts read `projects/transcoder-neuron/data/` through an absolute path at the top (`D = ...`). Set it to your clone. `vera_stats.py` also reads the env var `VD`.
