# Hugo signoff — theory-1020-megaprobe-after

Verdict: CONFIRMED.

What I rebuilt independently:

1. Score arithmetic from `data/scores.npz`, using my own code.
   - Code: `score_wide_check.py:21-60` for the 2nd-highest true threshold, catch rate and city bootstrap; `score_wide_check.py:127-154` for monitor loops and nulls.
   - Megaprobe pooled attacked catch: 0.7679, 95% city bootstrap CI [0.7427, 0.7914] (`score_wide_rebuild.json:260-264`). This confirms the title rule: point >= 0.50 and lower CI >= 0.40.
   - Baseline pooled attacked catch: 0.00783 [0.00112, 0.01734] (`score_wide_rebuild.json:255-259`).
   - MLP14 build-score pooled attacked catch: 0.00895 [0.00224, 0.01679] (`score_wide_rebuild.json:285-289`).
   - Label curve diff 800 -> 1400: 0.0358, so the prereg flatness stand-in fails (`score_wide_rebuild.json:291`).

2. Per-layer separation from score arrays.
   - Code: `score_wide_check.py:63-108` constructs eval twin pairs and computes d' retention.
   - Layer 8 retention: 0.612 [0.595, 0.627] (`score_wide_rebuild.json:45-50`).
   - Layer 14 retention: 0.131 [0.126, 0.136] (`score_wide_rebuild.json:3-8`).
   - Best checked upper layer was layer 18: retention 0.311 [0.302, 0.320] (`score_wide_rebuild.json:17-22`). This supports "no upper layer recovers" under the prereg recovery rule.

3. Raw-data rebuilds, not only score arithmetic.
   - MLP14 from raw `tok.npy` memmaps, with my own sklearn MLP: `raw_mlp14_knowledge.py:101-142`.
   - My MLP14 selected alpha 0.01 and got clean catch 0.985; pooled attacked catch 0.0157 [0.0050, 0.0291] (`mlp14_raw_rebuild.json:36-41`). This does not match Bruno's exact MLP, but it confirms that a clean-fit nonlinear layer-14 monitor does not rescue the attack.
   - Knowledge check from raw `lp.npy`, with token-prefix alignment fixed independently: `raw_mlp14_knowledge.py:146-190`.
   - Pair accuracy true continuation > false continuation: 1.0 clean and 1.0 for each attacked seed (`knowledge_rebuild.json:2-27`). This confirms the "still knows" gate.

4. Nulls.
   - Shuffled-label clean mega AUROCs: 0.475, 0.507, 0.500, 0.504, 0.480 (`score_wide_rebuild.json:62-68`).
   - Shuffled-label clean lin14 AUROCs: 0.476, 0.504, 0.474, 0.534, 0.480 (`score_wide_rebuild.json:55-61`).
   - Random raw L14 direction: AUROC 0.502 and catch 0.0 (`score_wide_rebuild.json:69-72`).

5. Novelty.
   - `novelty.md` says the broad multi-layer-obfuscation claim is not new. The defensible claim is fixed-text/original-probe/all-layer/untouched-upper-block measurement.

Caveat: my city-bootstrap CI for the megaprobe [0.7427, 0.7914] is slightly narrower than Bruno's [0.7422, 0.7942]. The decision is unchanged and both lower bounds are far above 0.40.

## Post-number check of PR #62 (blog head 6eb645b), 2026-10-20 15:40 CEST

Own code: `postcheck.py` (reuses only my helpers in `score_wide_check.py`) -> `postcheck.json`;
`null_tok18.py` (own raw refit from `tok.npy`) -> `null_tok18.out`.

Reproduced (post value = mine):
- Gap table, all 28 rows: original and fine-tuned gaps match to 2 dp at every layer; kept 61% L8, 13% L14, 21-31% at L15-28; all 14 upper layers beat L14; none has lower CI >= 0.5 (`postcheck.json` gap_table). L9 interval 52-56% with 300 boots vs post 53-56% (bootstrap noise).
- Megaprobe per run 67.6 / 74.3 / 88.4%; held-out false alarms 1.0 / 1.5 / 2.0%; clean 99.8%.
- Fit-seed refits 83.1% [81.0, 85.3] and 83.1% [80.6, 85.3].
- All-layer sentence-average probe: clean 98.7%, after 1.6% [0.6, 3.0].
- One-layer megaprobe arms: L18 46.1% (clean 99.0%), L22 12.9% (clean 55.9%).
- Country-token table (D3): every cell in post line 168 matches (L1-9 max 13.1%, L18 92.8% [90.9, 94.7], L21 93.9% [92.2, 95.5], L25-28 36.7-43.1%); clean >= 97.5% at L10-28; held-out false alarms max 3.0% (L21), L19 0.84% in every run.
- Own raw refit of the country-token probe (C=0.01, no CV): L18 92.84%, L21 93.90%, identical to d3.json.

Null for the country-token headline (`null_tok18.out`): 5 shuffled-label refits caught 0.06-6.4% after the fine-tune (L18 and L21); 5 random unit directions 0-9.9%. The 92-94% is not a fit or threshold artefact.

Wording issues found (not number errors in the headline):
1. Line 60 'a dent around the layer it was trained against, not a hole': false at the top; country-token catch L28 36.7% = L14 36.3% (Kasper #3882 raised the same).
2. Line 162 'line at the 1% point of the 596 held-out true statements ... 73.8%': the code uses the 5th-highest of 596 (4/596 = 0.67% false alarms). My variants: 5th-highest 73.8%, 6th 74.2%, 99% quantile 75.2%. Say '5th-highest' or give 74-75%; headline unaffected.
3. Lines 82/160 'moves by 6 points / about 10 points with the fit': fit spread over all megaprobes we have is 63% (independent skeptic fit) to 83% (refits), about 20 points.
