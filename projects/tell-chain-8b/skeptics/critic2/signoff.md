role: critic2
model: claude-opus-5-5
verdict: PASS

# critic2 sign-off: tell-chain-8b (Claude fallback, NOT cross-family)

The API critic2 (openai/gpt-5.5) hit its $2.44 cap before a verdict (room #32183). This sign-off is the Claude
fallback. It is disclosed as Claude and does not count as a cross-family check (Trust rule 4).

Read: PREREG.md (commit 92fa7c588, frozen rule section 3), DEVIATIONS.md, data/chain/*.json, data/base_px.json,
ref/parent_analysis.json at kran-research 72b2464f8; the post at blog f5a39ce and its deploy preview
https://deploy-preview-25--athanorscience.netlify.app/tell-chain-8b (200; title, banner, prereg foldout, Figure 1
PNG/og, figure data.json and the /lab card all served).

## Own rebuild (does not import analyze.py)

`rebuild_critic2.py` recomputes D(t) = mean(p_off) - mean(p_cue) - gap_base from the 64 per-prompt dev lists and the
untrained tail's lists in base_px.json, applies the PREREG pair rule at 0.10, floors at 8 and takes the seed-level
t-interval of mean ln R (df 4). Output `rebuild_critic2.out`.

| cell | rebuild | post / analysis.json |
|---|---|---|
| T of 35 learner records | 35/35 equal the stored T | equal |
| gap_base vs JSON | max abs diff 0.0 | - |
| censored (any arm) | 0 | 0 |
| R_D vs CTRL-D | 0.195 [0.165, 0.230] | 0.19 (0.17 to 0.23) |
| R_D vs FRESH-D | 0.212 [0.176, 0.255] | 0.21 (0.18 to 0.26) |
| status (both upper ends < ln 0.55 = -0.598) | CONFIRMED (hi_ln -1.469, -1.365) | CONFIRMED |
| PORT gate FRESH-D median T | 280 (band 266-326) | 280 |
| R_C vs CTRL-C / FRESH-C | 0.322 [0.262, 0.396] / 0.298 [0.248, 0.359] | 0.32 / 0.30 |
| clause 2 strictly falling R_B > R_C > R_D | 5/5 seeds | 5 of 5 |
| DOSE1-C / CTRL-C | 0.352 [0.273, 0.455] | 0.35 (0.27 to 0.45) |
| MAIN-C / DOSE1-C | 0.914 [0.723, 1.157] | 0.91 (0.72 to 1.16), not settled |
| MAIN-D / MAIN-C | 0.673 [0.541, 0.836] | 0.67 |
| trunk drift C / D | 0.927 / 1.086 | 0.93 / 1.09 |
| null cue E, max abs, main / trunk | 0.0143 (64 checks) / 0.0058 (110) | 1.4 / 0.6 points |
| D(0) largest | 0.0094 (main_D) | 0.009 |
| logit-unit T_D main / ctrl | 40-56 / 264-296 (15/15 = T_logit) | 40 to 56 / 264 to 296 |
| dev P(X|off) at t0, learners | 0.337 - 0.355 | 0.34 to 0.36 |

`cells_critic2.py` writes `rebuild_cells.json` (81 cells named as data/cells stems): 81/81 equal the team's cell files.

## Break attempts (`break_critic2.py`, `break_critic2.out`)

- Switch line 0.05 instead of 0.10: R_D 0.132 [0.096, 0.182] vs controls, 0.129 [0.096, 0.174] vs fresh. The yes gets
  stronger. Lines 0.15 and 0.20 cannot be read: learners leave at T + 8, so the curve stops before them (by design).
- Leave one seed out: upper ends 0.225-0.249 (controls), 0.253-0.280 (fresh), all below 0.55.

## Caveats (disclosed in the post; none changes the headline)

1. The frozen rule tests a threshold (0.55 at stage D), not a fixed factor per tell. The secondary read that
   separates the models (MAIN-C / DOSE1-C 0.91 [0.72, 1.16]) is not settled, and one same-size prior switch alone gave
   0.35 at stage C. The post says "though not by a fixed factor per tell" and does not claim 0.74 per tell.
2. Cue A is 8x larger than B-D, so count and size of earlier switches are confounded (post: Should you trust).
3. The step C -> D (0.67) follows 584 updates on cue C; no tail learned only C, so the drop cannot be split into
   "one more switch" and "longer training on a learned switch". The same-update controls got slower (drift 1.09).
4. One model, one cue size; the Lab rule for models above 2B was broken (laptop queue off), disclosed in Machines.
