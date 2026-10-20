# Vera (skeptic) sign-off — megaprobe-after (T3)

**Verdict: CONFIRMED (yes to the title), with two required disclosures (fit-seed spread; readout ablation).**

All scripts are my own (no import of build's code). Raw inputs: build's `data/acts/<state>/{tok,off,lp}.npy`,
`data/ref1019/{split,probe}`, and for the arithmetic check `data/scores.npz` + `data/results/results.json` (commit 8926df8).
My spec was fixed in room post #3746 before I saw any attacked score.

## 1. Raw data (T2)
- `vera_reextract.py` + `vera_check_acts.py`: 48 random eval statements x 4 states, re-run alone on CPU with my own
  model + peft load. Layers 1..27: min token cosine >= 0.9996, mean-pooled relative error <= 0.74% (CPU vs MPS bf16).
- Layer 28 in build's memmap is the last block output BEFORE the final RMSNorm (cos 0.60 to transformers'
  hidden_states[28]; cos 0.9995 after I apply the norm). Must be stated in the foldout.
- My L14 mean pool equals 1019 `clean_acts.npz` exactly (max diff 0.0); frozen probe: base AUROC 0.58297/0.57252/0.57105,
  catch 0.0067/0.0084/0.0084 = 1019 results.json.

## 2. Arithmetic from build's scores (`vera_rebuild.py`, `vera_rebuild.out`)
| monitor | pooled attacked catch | CI (my bootstrap, seed 1020) | lower CI over 5 boot seeds | clean |
|---|---|---|---|---|
| mega | 0.7679 | [0.7427, 0.7914] | 0.7399-0.7427 | 0.9983 |
| base | 0.0078 | [0.0011, 0.0173] | | 0.9597 |
| mlp14 | 0.0089 | [0.0022, 0.0168] | | 0.9916 |
| lin18 | 0.2886 | [0.2528, 0.3249] | | 0.9849 |
Per seed mega 0.676/0.743/0.884; eval false alarms 1.0/1.5/2.0%. Positive control: clean mega AUROC 0.99979 vs shuffled
0.475-0.507. Layer mass of build's mega: 72% off the top layer, ~0 on layers 1-14 (multi-layer check passes).
Label curve from scores: 200 0.540, 400 0.766, 800 0.804, 1400 0.768 -> clause 4 diff 0.036 > 0.03 (FAILED as build says).

## 3. Independent megaprobe (`vera_mega.py`, own PCA-64 features `vera_feats.py`, no HP search)
- Fit seeds 1020-1022: **0.730 [0.701, 0.758]**, clean 0.997. Other triples: 1023-1025 0.675 [0.643, 0.705];
  1026-1028 0.629 [0.598, 0.661]. All three clear 50% / lower CI 40%.
- **Fit-seed spread (~10 points) is larger than the city-bootstrap half-width (~2.5 points).** The headline CI does not
  include fit randomness. The post must say so; clause 4's 3.6-point gap is inside that spread.
- Shuffled labels (5): attacked catch 0.1-1.1%, clean AUROC 0.46-0.53 (`vera_mega_shuffle.out`).
- Label curve (mine): 400 0.532, 800 0.610, 1400 0.730.

## 4. Which part of the megaprobe does the work (`vera_mega_variants.out`, `vera_tokread.out`)
| readout (all fit on clean only, same 1% line) | pooled attacked catch |
|---|---|
| my attention probe, layer 18 only | 0.830 [0.805, 0.853] |
| my attention probe, layers 1-14 only | 0.317 [0.283, 0.351] |
| my attention probe, layer 14 only | 0.121 [0.098, 0.147] |
| linear, layer 18, 2nd-to-last token (end of country name) | **0.928 [0.909, 0.947]** |
| linear, layer 14, 2nd-to-last token | 0.363 [0.338, 0.387] |
| linear, layer 18, last token / max over tokens / mean | 0.338 / 0.174 / 0.289 |
Reading: the catch comes from reading the country token at a layer above 14. Breadth over layers is not needed: a
one-layer linear probe at the right token beats the megaprobe. The post may answer the title "yes", but must not imply
that reading every layer is what saves the monitor. (These readouts were chosen by me after the prereg; they are
diagnostics, not the status.)

## 5. Per-layer (`vera_lin.py`, C = 0.01 fixed; identical to build's chosen C for l >= 2)
r_14 0.13 [0.13, 0.14]; r_15..28 0.21-0.31 (max l18 0.31 [0.30, 0.32]); no layer recovers (lower CI >= 0.5).
Clause 1 rests on l8 alone, clean d' 1.02, just over the 1.0 scoring gate (r 0.61). Random unit directions at l8 give
|d'| <= 0.08, so l8's d' is real, but a slightly weaker clean probe would make clause 1 untestable.

## 6. Knowledge (`vera_know.py`)
Affirmative eval pairs: P(true continuation) > P(false) in 298/298 pairs, clean and every seed. 'Still knows' rule holds.

## 7. Added after D2 (15:23 CEST), `vera_mega_variants.out`
My one-layer attention probe, fixed HPs, clean-fit: layer 18 = 0.830 / 0.716 / 0.743 over 3 fit-seed triples (clean
0.995-0.998); layer 22 = 0.873 [0.850, 0.895] (clean 0.990). Build's D2 arm A (46.1% at l18, 12.9% at l22 with clean
55.9%) fails its clean control, so it is an under-fit, not evidence for breadth. Build's arm B (all 28 layers,
sentence average, linear) 1.6% agrees with my per-layer sentence averages (<= 28.9%).

## 8. Figure (T6) review
First version: numbers match (`vera_lin.json`, <= 0.0056); text failed on '1% false alarms' (eval 1.0-2.0%) and on
'No single layer keeps more than a third' (true only for sentence-average probes). See room #3807.

novelty.md: the 2606.09411 row was corrected after Iris #3811 (that attack summed its loss over all layers).

## 9. Displayed equations
- d' (post foldout): my own computation in `vera_lin.py` (dprime in `vera_core.py`) gives build's per-layer d' to <= 0.0056
  (layer 1 uses another C).
- Megaprobe score equation: `vera_eqcheck.py` evaluates the displayed formula in float64 numpy on build's saved fit
  (`data/fits/final_mega.pt`, `pca_clean.pt`; z = scale * V^T (x - mu), scale = 1/sd) for 40 statements: max |diff| to the
  saved scores 0.0019 (score sd 3.84; the saved scores used fp16 z). The formula in the post is the one that was run.
