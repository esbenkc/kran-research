# Vera (skeptic) sign-off — warning-override

**SIGN-OFF: REJECTED (No to the title), confirmed by my own rebuild.** The prediction FAILED (both parts).

Inputs: build's raw run logs only (`projects/warning-override/raw/<model>/<cell>/<seed>.jsonl`, 800 runs at
kran-research 7ca1eff6). My own tagger `vera_score.py` (written from PREREG s2; build's scorer.py and analysis.py
not read) and my own analysis `vera_check.py`, `vera_robust.py`. CPU, < 1 s each, ~30 MB.

## Headline (PREREG s3, flag-present cells, 5 models equal weight, within-model run bootstrap B=10,000, 90% CI)

| number | mine | build (results/summary.json) |
|---|---|---|
| unwarned: key / flag / any | 0.685 / 0.500 / 0.950 | 0.685 / 0.500 / 0.950 |
| warned: key / flag / any | 0.215 / 0.070 / 0.255 | 0.215 / 0.070 / 0.255 |
| F (warning's rise in flag use) | -0.4300 [-0.4800, -0.3800] | -0.430 [-0.480, -0.380] |
| K (warning's cut in key use) | +0.4700 [+0.4050, +0.5350] | +0.470 [+0.405, +0.535] |
| status (edge +0.0725) | F upper -0.38 < +0.0725 -> REJECTED | REJECTED |
| A (total cut) | +0.6950 [+0.6400, +0.7450] | +0.695 [+0.640, +0.745] |
| R (pooled Haldane log-odds ratio) | 1.9250, joint CI [1.6921, 2.2554] | 1.925 [1.692, 2.255] |
| prediction | failed (R CI lower 1.69 >= 0.5; F CI upper -0.38 < 0.15) | failed |
| per-run tag diff (key, flag, any) | 0 of 800 runs differ | — |

## Null and robustness (vera_check_counted.out, vera_robust.out)

- **Warning-label permutation within model** (10,000 shuffles, flag-present runs): null F 5-95% = [-0.07, +0.07];
  observed -0.43; P(null <= -0.43) < 1e-4.
- **R in flag-absent cells = 1.0000 exactly** (any = key when the flag is not offered; 0 invented flag calls).
- **Seed halves**: even seeds F -0.47, odd seeds F -0.39 (both No).
- **Leave-one-model-out**: all five read No (F -0.33 to -0.54, every CI upper < -0.27).
- **Model-cluster CI** (models then runs): F [-0.670, -0.185] -> still No.
- **Other definitions**: exposed-only F -0.430; runs that saw the flag line -0.430; clean flag (no key in any flag
  call) -0.225 [-0.275, -0.175]; flag-only runs (never the key) -0.225; flag or near-miss spelling -0.430
  (0 near-miss spellings in 800 runs). Every variant says No.
- **Per model**: F = Haiku -0.70, Gemini 3.1 Flash-Lite -0.50, Gemini 3.5 Flash-Lite -0.825, GPT-5.4 nano
  -0.125, GPT-5.4 mini 0.00 (mini used the flag in 0 of 80 runs, though it saw the line in 78). No model rose.

## Caveats I found (for the post, not status-bearing)

1. **Anchor only partly replicates.** Flag-absent cells: key 0.920 -> 0.305 here vs honeypot-talk's obvious record
   0.873 -> 0.203. Warned key use is 10 points higher (61/200 vs 71/350, Fisher two-sided p = 0.009), carried by
   GPT-5.4 nano (0.425 vs 0.157) and GPT-5.4 mini (0.700 vs 0.543). The new usage line in every denial is the only
   design change, so it may have weakened the warning a little. The post should quote both numbers.
2. **No in-scope route exists.** Less straying equals giving up the blocked step: the bucket size was obtained in
   190/200 unwarned vs 51/200 warned flag-present runs.
3. **Bundled calls.** 47/200 unwarned flag-present runs passed the key and the flag in the same call; the clean-flag
   row (-0.225) is the conservative number for 'moved to the flag'.
4. **Scope.** One generic identify-only warning, one flag label, five small API models. Novelty in novelty.md.

Power was simulated before seed 0 (vera_power1.py, vera_power2.py): the observed F is 50 points below the edge, so
the status does not sit near a boundary.
