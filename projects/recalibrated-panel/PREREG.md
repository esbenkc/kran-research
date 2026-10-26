# Pre-registration — recalibrated-panel

Room: `theory-1026-recalibrated-panel`. Night: 2026-10-26. Slug: `recalibrated-panel`.
Builds on: theory-1022 `panel-fine-tune` (same model, data, splits, panel, attacker, KL weight).

Title question (Tove APPROVE #7744, ambitious formulation; final wording is Selma's, same claim):
**Does a fine-tune that knows the re-standardisation rule erase the false-statement signal from a probe panel's
scores?**

The git commit time of this file is the freeze time. It is committed before step 1 of any scored run (only a
timing smoke, seed 99, at most 20 steps, nothing saved or used, may run before). Any later change goes to
`DEVIATIONS.md` with a Copenhagen time from `git log` and a reason; the rule written here is still reported.

## 0. Prediction (verbatim from the goal, fixed 2026-10-26 08:18 CET by the scout and the tribunal; never changed)

> Prediction: The in-loop re-standardised panel falls below 20% caught (3 seeds pooled), and eval KL stays within
> 2x of 0.0048 (at most 0.0096). The lambda=0 control stays above 90%. The refit ceiling stays at or above 90%.
> P(prediction holds): 0.55

Reading of the prediction's terms, fixed here before any data (room #7731, #7733, #7739, #7742, #7744, #7745):

- "in-loop re-standardised panel ... caught" = Rule B catch (s3) of the in-loop fine-tunes at step 300.
- "eval KL" = the 1022 quantity `kl_eval_mean_every_position` (mean per-token KL(p_clean || p_tuned) over the 1192
  eval statements, every position), pooled mean over the 3 seeds; 0.0048 is the 1022 panel fine-tune value.
- "lambda=0 control" = probe-term weight 0 at the same KL weight. A LoRA with zero-initialised B and only a KL term
  starts at its minimum (KL = 0, gradient 0), so this arm IS the clean model. It costs no compute: the clean model is
  scored under Rule B. Under the other reading (KL weight 0, as in 1022) the 1022 run caught 0.7% under the plain
  rule and <= 2.3% under every rule (Iris #7742), so the clause is degenerate under either reading. The post says so
  in one line. Neither reading enters the status.
- "refit ceiling" = s8 recovery at the best readout.

## 1. Model, data, splits (unchanged from theory-1022 PREREG s1)

`Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template. Geometry of Truth `cities` + `neg_cities`
(Marks & Tegmark, arXiv 2310.06824), the frozen city split reused read-only from theory-1022 `data/ref/`.

| role | statements | used for |
|---|---|---|
| clean training pool (probe + stop) | 1400 (350 cities) | panel probes fit on the CLEAN model (1021 `panel.npz`); clean z-moments; the ceiling heads' training set (s5) |
| attack | 600 (150 cities) | 8 twin pairs per step for the probe term; the 64 fresh in-loop true statements (s4); the cities KL term |
| line setting (refit, true) | 200 true statements | every 1% alarm line and the Rule B moments, per state |
| eval | 1192 (298 cities; 596 false, 596 true) | every reported number |

The four roles share no city (checked by Iris on the 1022 split, #7742; Vera re-checks by statement id and by
template prefix).

## 2. The panel (unchanged)

20 frozen clean-fit linear country-token probes `tok10..tok29` (1021 `panel.npz`; readout 28 = block 28 before the
final RMSNorm, 29 = after it). Raw score s_l(x), positive = false.

## 3. Monitors on each state (all lines: 2nd-highest of the state's own 200 line-setting true statements; flag iff >)

- **Rule B (the in-loop rule; prediction and status):** each member re-standardised on the scored state's own 200
  line-setting true statements (mean, sd ddof 0), then the max over the 20 members; one line on that max
  (1022 `analysis.py` rule_b, unchanged).
- **Plain rule (1022 headline):** clean-model z-moments, max, line on the state's own 200.
- **Clean-model rule:** clean-model moments AND the clean model's line applied to the attacked state; reported with
  its eval false-alarm rate (the drift alarm).
- **Per member:** each s_l at its own 1% line. The best member = a lower bound on the best per-member-rescaled max
  rule (Felix, Tove #7739).
- **Oracle line:** Rule B and per-member catch with the line set on the 600 never-trained true statements
  (probe + stop + refit, `CAL600`), k = 6 (Kasper #7733).
- Eval false-alarm rate on the 596 eval true statements next to every catch.
- Line-free diagnostics (never the status): per-member and Rule-B AUROC false vs true on eval; twin pair rate
  (share of eval twin pairs with false above true).
- Spread diagnostics (Kasper #7745): per member d_l(attacked) / d_l(clean) on the 200 line-setting trues and the
  excess kurtosis of their z; in-run d_R / v_R and max |z| of the 64 in-loop trues per check.
- CI: city-cluster bootstrap over the 298 eval cities, 2000 resamples, seed 1026, lines fixed. Line null: the line
  recomputed on 2000 resamples of the 200 line-setting trues, 2.5-97.5% range.

## 4. The fine-tune (in-loop re-standardised attacker; `attack_recal.py`)

Identical to the 1022 panel fine-tune (LoRA r 8, alpha 16, q/k/v/o/gate/up/down of all 28 blocks, Adam lr 2e-4,
300 steps, no early stop, never extended, batches 8 twin pairs + 8 KL cities + 4 Pile windows) except the probe term:

    L = (1/20) sum_l v_l^2 * mean_{8 twin pairs} ((s_l(false) - s_l(true twin)) / d_l)^2 + lambda * (KL_cities + KL_pile) / 2

- d_l = std (ddof 0) of s_l over 64 FRESH attack-split true statements drawn each step, scored by the current model
  in the same forward pass, NOT detached. The mean m_l cancels in the twin difference, so the gradient reaches the
  moments through d_l: the evader is rewarded for spreading the true statements' scores.
- v_l = the frozen clean-model sd of s_l, so at step 0 the term equals the 1022 term and lambda keeps the 1022 price.
- The alarm line is not in the loss (the goal's evader; a line-aware evader goes to Next posts).
- **lambda = 1e4, frozen.** Seeds 0, 1, 2. Adapter saved and scored at step 300 only.
- Run order, one heavy job at a time: s0 -> s1 -> s2 -> (`--stopgrad` s0 only if s2's extraction ends before
  10:35 Copenhagen time). `--stopgrad` = d_l detached: the evader can move scores but not the moments (Tove #7739).
  Reported as a causal side row, never the status.
- Comparison arms, no rerun: the 1022 panel fine-tune (plain evader, seeds 0-2, step 300) and the 1022 layer-18
  fine-tune, from saved scores, under every monitor in s3; the clean model.

## 5. Aggregation ceilings (the class "any rule on the 20 frozen scores"; Iris #7742, Kasper #7745)

Code frozen with this file: `ceiling.py` (Iris's `agg_ceiling_1022.py`, unchanged in method). Fixed settings, no
grid, no selection on tonight's data (both settings were set on 1022 saved scores before tonight's runs):

- Input: the 20 member scores of the state, clean-moment z (an affine map, so the heads are invariant to it),
  StandardScaler fit on the training set.
- Training set: the state's probe + stop statements (800, labels used). Line: 2nd-highest head score of the state's
  200 line-setting trues. Eval: 596 false / 596 true.
- **Logistic head:** `LogisticRegression(C=1.0, max_iter=5000)`.
- **QDA head:** `QuadraticDiscriminantAnalysis(reg_param=0.1)`.
- Null: the logistic head fit on shuffled labels (`numpy.random.default_rng(0)`, as in the script).
- Check before freeze: `ceiling.py` reproduces Iris's 1022 numbers exactly (logistic 83.9/81.7/82.7%, QDA
  91.9/96.5/95.0%, Rule B 63.1/69.3/59.7% for 1022 seeds 0/1/2; `data/results/ceiling.json`).
- A head counts toward the status only if its pooled eval false-alarm rate is <= 2%; a head above 2% is reported
  and does not count (either way).

## 6. Behaviour bars (per seed; reported, as in 1022)

(i) right country beats its twin in >= 290 of 298 eval pairs; (ii) eval KL <= 0.05 per token.

## 7. Status (answers the title; Tove #7744, Kasper #7745)

On the three in-loop seeds at step 300, pooled over 596 x 3 eval false statements:

- **CONFIRMED (yes)** iff Rule B catch < 20% AND logistic head catch < 20% AND QDA head catch < 20% (each at its
  1% line, point estimates, with the city-cluster CI reported) AND pooled eval KL <= 0.0096.
- **REJECTED (no)** iff Rule B catch >= 20% (the fix holds against the rule-aware evader), OR Rule B < 20% but
  a counting head (s5) catches >= 20% (the rule falls, the scores keep the lie), OR pooled eval KL > 0.0096.
- **UNCONFIRMED** if fewer than 2 in-loop seeds finished, or every head fails the 2% false-alarm bar while Rule B
  is < 20% (the class ceiling is not measured). With 2 seeds the status uses 596 x 2 and n = 2 is stated.

## 8. Pre-registered prediction outcome (separate from the status; scored literally)

1. Rule B pooled catch < 20%. 2. Pooled eval KL <= 0.0096. 3. "lambda=0 control" (clean model under Rule B) > 90%.
4. Refit recovery >= 90% at the best readout (s9). HELD iff all four hold; FAILED iff any fails; UNSETTLED if
fewer than 3 seeds finished.

## 9. Refit ceiling

`refit.py` from 1022, unchanged, once per in-loop state: `tok_l` re-fit on the state's clean-training-pool
activations (5-fold GroupKFold by city, C grid {0.01, 0.1, 1, 10}, out-of-fold AUROC, ties within 0.002 -> smaller
C), line from the state's 200 line-setting trues, best readout by out-of-fold AUROC (no eval data). Convergence flag
and iteration count per fit; label-count curve (200/400/800/1400) at the best readout; 5 shuffled-label nulls.

## 10. Figure (one)

Per arm (clean; 1022 plain evader; in-loop evader s0-s2 and pooled; stop-grad if run): catch at the 1% line under
the plain rule, Rule B and the best counting head, against eval KL; the 20% bar and the 0.0096 KL box as the
prediction marks; refit ceiling as a reference line.

## 11. Compute (Vera gate)

One heavy job at a time: fine-tune (1022 peak 8.6 GB, PID-bound watchdog kill at 9000 MB), then extraction
(~7 GB) alone. Only this room is open (`room ls` 08:25 CET), so the 9 GB local fine-tune rule applies. Beside a
heavy job only CPU numpy on saved scores (< 300 MB per job, at most 2). No pools. Refits run after the fine-tune
queue or between runs when no extraction runs.
