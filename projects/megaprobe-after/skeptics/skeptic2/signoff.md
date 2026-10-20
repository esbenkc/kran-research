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
