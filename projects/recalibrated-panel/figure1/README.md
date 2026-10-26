# Figure 1 (Owain)

Builds `blog.kran.ai/assets/figures/recalibrated-panel-fig1/figure1.html` from saved scores only.

- `pick.py`: Rule B (= `analysis.py rescaled()`), twin lookup within sentence form.
- `pick2.py`: mechanical example pick. Qualifies a false statement that the clean model's panel and classifier catch, that the
  in-loop panel misses and the classifier catches in all 3 seeds, that the plain-1022 classifier catches in all 3 seeds, and whose
  true twin no rule flags. Output: `pick_neg_3seeds_logit.txt` (75 negated-form lies qualify; 6 affirmative). Kobe = index 2250.
- `gen2.py` + `fig_cfg2.json`: the figure. Classifier = logistic head of `data/results/ceiling.json` (agg_ceiling); every per-seed catch
  is asserted equal to `ceiling.json`. Piles: the middle run (in-loop seed 2, plain-1022 seed 0).
- `coldread.py` / `coldread_final_v4.json` (final wording; `coldread_final_v2.json`, `coldread_g9.json` = earlier wording): PNG-only cold read, 3x claude-haiku-4-5 + 3x gemini-flash-latest, 6/6 question + "No".

Run from `projects/recalibrated-panel`: `FIG1_BLOG=<blog clone> ../../.venv/bin/python figure1/pick2.py`, then `figure1/gen2.py`.
Sources: data/scores/{clean,recal__lam10000__seed0..2}.npz, data/prior1022/scores/panel__lam10000__seed0..2.npz,
data/ref/split.json, data/fits/zmoments.json, data/results/ceiling.json.
