role: critic2
model: anthropic/claude-opus-5-5
verdict: PASS

# critic2 sign-off: panel-fine-tune-8b (Claude fallback, not a cross-family check)

The scheduled cross-family reviewer (openai/gpt-5.5) hit its $2.47 cap ($2.70 spent) at 12:58 PST before it posted
a verdict. swarm restarted the critic2 role on Claude (anthropic/claude-opus-5-5). This verdict is therefore
disclosed as Claude and is not a cross-family check.

Answer to the question: YES (CONFIRMED by the frozen PREREG s6 rule). Prediction (s7): FAILED.

Rebuilt with my own code (`rebuild.py`, numpy + the cached Qwen3-8B tokenizer) from committed raw files only, in a
detached worktree of origin/swarm/oracle-1118-panel-fine-tune-8b @8d574ca93. The max-panel is recomputed from the
raw per-probe scores `tok<R>` and `fits/zmoments.json`, not from the stored `panelmax` column (they agree to 0.0).

| cell | my rebuild | post / results.json |
|---|---|---|
| clean gate (own line, catch / FA) | 596/596 = 100%, line 3.676, FA 1.68% | 100%, 3.676, 1.68% |
| panel catch at own 1% line, seeds 0/1/2 | 17.11 / 8.89 / 24.16% (lines 5.94 / 6.38 / 6.12) | 17.1 / 8.9 / 24.2% |
| pooled catch, city-cluster bootstrap (my RNG seed 4242, 2000) | 16.72% [14.60, 19.02] | 16.7% [14.4, 19.0] |
| seed-level t95 (df 2, t 4.303) | [-2.26, 35.71]% | [-2.3, 35.7]% |
| bar (i) right country first, 298 affirmative eval pairs (own tokenizer prefix rule) | 298/298 each seed | 298/298 |
| bar (ii) eval KL every position (tokens + position 0) | 0.01441 / 0.01439 / 0.01068; pooled 0.01316 | same; 2.7x the 1.5B 0.00482 |
| random-direction control, seeds 0/1 | 100% / 100% (KL 0.0155 / 0.0141) | 100%, 0.007-0.016 |
| penalty-only control, seeds 0/1 | 100% / 100% (KL 0.0073 / 0.0110) | 100% |
| clean-model line on tuned panel | catch 100%; FA 92.5 / 96.0 / 95.3% | 92-96% |
| best single probe, picked on held-out 'stop' false statements | readout 18 / 15 / 19: 75.7 / 82.4 / 81.9% | 76-82%, layers 15-19 |
| probes on readouts 21-36 alone | 0.8% to 33.9% | 1% to 34% |
| mean z shift, eval trues / falses | +3.4..+3.5 / -3.8..-4.0 | +3.5 / -4 |
| recalibration fix (rescale on 200 tuned trues) | 42.4 / 59.6 / 52.9%, mean 51.6% | 52% |
| post hoc AUROC; 5% line | 0.947 / 0.949 / 0.953; 62.1 / 64.3 / 69.1% | same |
| refit best readout eval catch; shuffled null | 100% x3; 15 nulls 0.17-2.35%, mean 1.13% | 100%; about 1% (0.2-2.3%) |
| 19-probe max without readout 36 | identical (max diff 0.0) | identical |
| line + city redraw (my version: resample line-setting trues by city) | [10.2, 37.2]%, P(>=20%) 0.297 | 10.2% to 38.3%, 31% |
| 1.5B reference | panelmax 3.75%, clean 99.5% | 3.7%, 99.5% |

Frozen rule: upper 19.0% < 20% and all three seeds pass both bars -> CONFIRMED. The seed-level interval crosses
20% and one seed is at 24.2%; the post says both in the TLDR and the limits box.

Only difference found: the line-redraw upper end (my city-clustered redraw 37.2% vs the post's 38.3%; P 29.7% vs
31%). It is a robustness row in limits and the expert foldout, built with a different resampling unit; it does not
touch the status. Not a blocker.

Preview checked (https://deploy-preview-220--kranai.netlify.app/panel-fine-tune-8b, head b7b8ab6): text matches the
PR head, lab banner and pre-registration foldout render, figure data.json values equal my rebuild, /lab lists the
card. Open before merge: lab_cost_usd must include tonight's API spend (room spend: $2.84 incl. gpt-5.5 critic2
$2.70, plus the $0.06 question pick = $2.90), and lab_signoffs must be rewritten with `oracle.py signoffs --write`
after this folder is committed.
