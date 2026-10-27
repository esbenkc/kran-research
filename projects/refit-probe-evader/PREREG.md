# Pre-registration — refit-probe-evader

Room: `theory-1027-refit-probe-evader`. Night: 2026-10-27. Slug: `refit-probe-evader`.
Builds on: theory-1022 `panel-fine-tune` and theory-1026 `recalibrated-panel` (same model, data, splits, panel
readouts, LoRA, optimiser, steps, KL weight). Planner: Felix. Reviewers: Tove (strategist), Kasper (critic).

Title question (Selma #8352, keeps the price clause per Tove #8348):
**Does a fine-tune blind probes retrained with labels on it, at the output price that LEACE predicts?**

The git commit time of this file is the freeze time. It is committed before step 1 of any scored run. Only two
things may run before it: the LEACE price script (forward passes on the clean model, no fine-tune), and a timing
smoke (seed 99, at most 20 steps, nothing saved or used). The LEACE prices are added as v1.1 BEFORE step 1 of R1.
Any later change goes to `DEVIATIONS.md` with a Copenhagen time from `git log` and a reason; the rule written here
is still reported. v1 = 8467ee5 (15:04:13 CET). v1.1 = this text: registered price values (s4), and s2/s3 aligned to
the code before any run (Kasper #8410, Felix #8411: in-loop/M1 C per readout from `data/fits/C.json`; every max-panel
z-scored by the 200 line-setting trues; stress-probe settings). No status rule was loosened.

## 0. Prediction (verbatim from the goal, fixed 2026-10-27 14:52 CET by the scout and the tribunal; never changed)

> Prediction: At the base KL weight the refit max-panel falls below 20% caught at 1% FPR (3 seeds pooled) with
> eval KL within 2x of 0.0048 (at most 0.0096). The realised KL lands within 3x of the all-layer LEACE floor. The
> 10x-higher KL weight keeps catch above 50%. Fresh held-out-label probes at the end also stay below 30%.
> Right-country ranking stays >= 290/298.
> P(prediction holds): 0.45

Reading of the terms, fixed here before any data (room #8329-#8362):

- **"LEACE floor" is read as "the LEACE price".** All 7 phase-0 takes agree (#8329-#8335): LEACE (Belrose et al.,
  arXiv 2306.03819) is the least-squares affine eraser (smallest mean squared change to the activations under a
  covariance norm), not the smallest-output-change edit. A fine-tune can in principle hide the concept for less.
  So the price is a prediction, the 3x band is two-sided, and the words "floor" and "lower bound" are not used for
  it anywhere in the post.
- **"refit max-panel"** = monitor M1 (s3). **"Fresh held-out-label probes"** = monitors F1 and F2 (s3).
- **"eval KL"** = `kl_eval_mean_every_position` of `extract.py` (mean per-token KL(p_clean || p_state) over every
  position of the 1192 eval statements), pooled = mean over the base-weight seeds. 0.0048 = the 1022 panel
  fine-tune value.
- **"10x-higher KL weight ... catch"** = M1 catch of arm R2 (lam 1e5).
- **"Right-country ranking"** = `metrics.py` pairs rule, per seed, 298 eval pairs.

## 1. Model, data, splits (unchanged from theory-1022 / 1026)

`Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template. Geometry of Truth `cities` + `neg_cities`
(Marks & Tegmark, arXiv 2310.06824); the frozen city split `data/ref/split.json`, read-only.

| role | statements | used for |
|---|---|---|
| probe + stop | 800 | fresh probes F1/F2 (s3); LEACE selection fit (s4) |
| attack | 600 | in-loop retrained probes and twin pairs (s2); M1; LEACE selection score (s4) |
| fit pool = probe + attack + stop | 1400 | the registered LEACE eraser (s4) |
| line setting (`refit` split) | 400 (200 true) | every 1% alarm line, per state |
| eval | 1192 (298 cities) | every reported number |

Roles share no city (1022 check; Vera re-checks by statement id).

## 2. Arms (serial; one heavy job at a time; plan #8362)

Common: LoRA r 8, alpha 16, all 7 projections, all 28 blocks, Adam lr 2e-4, 300 steps, batch 16 positive (8 twin
pairs) + 8 KL statements + 4 pile windows, KL weight lam (1022 loss with the frozen probes swapped as below).

| run | arm | in-loop target | lam | seed |
|---|---|---|---|---|
| R1 | retrained-probe fine-tune | 20 logistic probes (readouts 10..29, country token, StandardScaler + L2 logistic, lbfgs, C per readout = the clean panel's CV-picked C in `data/fits/C.json`; v1.1, Kasper #8410, Felix #8411) refit on the CURRENT model's 600 attack-split statements with true labels at steps 0, 50, ..., 250; loss term = mean over readouts of the squared twin gap of the current probes | 1e4 | 0 |
| R2 | same | same | 1e5 | 0 |
| R3 | same | same | 1e4 | 1 |
| R4 | same | same | 1e4 | 2 |
| N | shuffled-label fine-tune (price control) | same schedule, but every refit uses ONE fixed permutation (seed 0) of the 600 attack labels | 1e4 | 0 |
| G | mean-gap fine-tune (side arm) | per readout R: d_R = W_R (mean_false - mean_true) of the batch country-token states, W_R = the whitening of the SELECTED LEACE setting (s4), fixed from the clean model; loss term = mean over R of mean(d_R^2) | 1e4 | 0 |

Run order R1, R2, R3, R4, then slots 5 and 6. Branch rule (Tove #8364; uses only R1's own result, not a seed
selection): after R1 is scored, if R1's F1 AND F2 both catch < 30%, slot 5 = N and slot 6 = G; otherwise slot 5 = G
and slot 6 = N. Slot 6 starts only if it can start by 10:15 PDT (18:15 CET). Compute stops at 10:50 PDT (18:50 CET).
The base arm needs at least 2 of R1/R3/R4 for any status other than UNCONFIRMED. R2, N and G are n=1 arms and the
post says so.

Trajectory (figure only, no status role; Tove #8371, Kasper #8372, Bruno #8382): at steps 0, 50, ..., 300 each run
logs (a) catch of the current retrained max-panel on a fixed held-out slice of 200 eval statements (100 cities, true
and false; never fitted), line from the 200 line-setting trues, and (b) `kl_eval_mean_every_position` on the same
slice. The step-300 point is replaced by the full-eval numbers; Vera checks slice vs full at step 300.

## 3. Monitors on each final state (step 300; every line = 2nd-highest of that state's 200 line-setting trues)

- **M1, retrained max-panel (the deployed refit monitor).** The R1 in-loop procedure run once more at step 300 on the
  final model: 20 probes on the 600 attack-split statements, true labels, C from `data/fits/C.json`.
- **F1, fresh max-panel (held-out labels).** 20 probes fit on the final model's probe+stop (800) statements by the
  `refit.py` rule (StandardScaler + L2 logistic, C from {0.01, 0.1, 1, 10} by 5-fold GroupKFold-by-city OOF AUROC,
  ties within 0.002 -> smaller C); F1 = max over readouts.
- **F2, best fresh probe.** The single F1 readout with the highest OOF AUROC (ties -> lower readout). No eval data
  picks it. Code: `monitors.py`, `evaluate.py`.
- Every max-panel (M1, F1, trajectory) z-scores each readout by the mean and std of that state's 200 line-setting
  trues before the max (v1.1: the 1026 rule, out of sample for every probe; replaces v1's attack-trues and
  probe+stop-trues z, Kasper #8410).
- Catch = share of the 596 eval false statements above the line. Pooled = mean of per-seed catch; 95% CI =
  `metrics.boot_catch` (city-cluster bootstrap, 2000 draws, rng 1021).
- AUROC of M1, F1, F2 is reported beside every catch (Vera #8383: catch at 1% false alarms is a tail metric; an
  eraser fit on 600 rows left fresh-probe AUROC 0.81 at 0.2% catch). Secondary label, no status role: a state is
  "fully blinded" only if F1 and F2 catch < 30% AND their AUROC <= 0.60. If the status is CONFIRMED but the state is
  not fully blinded, the post says in the TLDR that a weaker linear signal survives below the 1% tail.
- Stress rows, never in the status: a quadratic probe (QDA on the top 64 principal components, reg 0.5) and a
  2-layer MLP probe (256 hidden) fit fresh on probe+stop at F2's readout (Vera, Iris: equal means can leave a
  variance tail or a +v/-v re-encoding); a shuffled-label fresh probe at F2's readout (5 permutations, must read ~1%).
- Known-answer cells on the CLEAN model: F1/F2 on the clean model (ceiling); F1/F2 on the clean model with an
  all-layer LEACE eraser of the selected setting fit on the ATTACK split (600) only, the fine-tune's data access
  (Vera #8383), fresh probes on probe+stop, scored on eval (the gate in s6); and the same with the registered
  1400-pool eraser (reported; partly in-sample).

## 4. The LEACE price (computed before R1; values frozen in v1.1)

**Registered values (v1.1, from `data/leace/price1.json`, Bruno #8419, before step 1 of any scored run):**

| setting | selection fit | OOF attack AUROC (best readout) | all-layer price, 1400 fit, seq 10..29 |
|---|---|---|---|
| plain pseudo-inverse | probe+stop 800 | 0.801 (readout 18) | 0.02140 nats/token |
| **Ledoit-Wolf (chosen)** | probe+stop 800 | **0.765** (readout 22) | **P = 0.006145 nats/token** |

Chosen by the s4 rule: |0.765 - 0.5| < |0.801 - 0.5| by 0.036, outside the 0.01 tie band. **P = 0.006145**; band
[P/3, 3P] = **[0.00205, 0.01844]** nats/token. Vera's independent plain rebuild (own CPU fp32 code): 0.0209
(Bruno's plain 0.02140, 2.4% apart). Note: both 800-fit erasers leave the attack split readable out of sample
(AUROC 0.77-0.80), so the s6 gate is a real test.

- Eraser: LEACE (Belrose et al. 2306.03819; reference code EleutherAI `concept-erasure`, LeaceFitter), binary concept
  (false = 1), fit on the CLEAN model's country-token states.
- **Sequential** across readouts 10..29 (Kasper #8334, Vera #8335): fit at readout 10, apply it, recompute readout 11
  with the readout-10 eraser active, fit at 11, and so on through 29 (29 = post final RMSNorm, the logits' input).
- The edit is applied at the **country token only** (Kasper #8342), inside the bf16 forward (eraser math in fp32,
  cast back to bf16). Positions before the country token are untouched.
- **Registered price P** = `kl_eval_mean_every_position` of the clean model with all 20 sequential erasers applied,
  versus the clean model, on the 1192 eval statements: the same function, statements and positions as every run.
- Also reported (no status role): each readout alone; readouts 10..28 without 29, with the readout-29 fresh-probe
  catch on that 10..28-erased clean model (Kasper #8384: RMSNorm can make 29 readable again); all readouts fit
  independently; the every-position variant (eraser applied at every position); a random-direction rank-1 control
  edit of the same size (Bruno #8382); optional label-aware shift line (Iris #8330, #8355), labelled "second-order
  price of a class-constant shift with the rest of the model frozen", never "bound" (Vera #8347, #8361; Kasper #8351).
- **Setting selection (room #8374, amended #8377 before any calibration run).** Grid: {plain pseudo-inverse
  whitening, Ledoit-Wolf shrinkage}. Each setting is fit on probe+stop (800) only; on the clean model with that
  all-layer eraser, a logistic probe (C grid as F1) is scored by 5-fold GroupKFold-by-city OOF AUROC on the attack
  split (600) at its best readout. Pick the AUROC closest to 0.5; ties within 0.01 -> the smaller all-layer price.
  The chosen setting is then refit on the full 1400 fit pool for P. Eval is not used.

## 5. Dtypes (every stage)

Model forward and backward bf16 on MPS (runs, extraction, price). KL from fp32 log-softmax of the logits
(`extract.kl_rows`). Probes and LEACE fits in float64 on fp32 copies of the fp16-saved country-token states.

## 6. Status of the title (base weight = R1, R3, R4 pooled)

CONFIRMED (yes) iff all hold, with at least 2 base seeds complete:
1. fresh probes blinded: F1 < 30% AND F2 < 30% pooled;
2. retrained panel blinded: M1 < 20% pooled;
3. outputs kept: pooled eval KL <= 0.0096 AND every base seed ranks the right country first in >= 290/298;
4. at the predicted price: pooled eval KL in [P/3, 3P];
5. not machinery (Tove #8348 b): the shuffled-label fine-tune N has eval KL < 0.5 x the pooled base eval KL.

REJECTED (no) iff at least 2 base seeds are complete and any of 1-5 fails. The post names which:
"holds" (1 or 2 fails), "outputs broken" (3), "falls off-price, over / under" (4), "price is LoRA machinery" (5).

UNCONFIRMED iff fewer than 2 base seeds complete; or Vera's or critic2's rebuild of a status cell differs by more
than 2 catch points or 5% of a KL; or N did not run while 1-4 all hold.

Known-answer gate: if F2 on the clean model with the attack-split-fit eraser (s3) catches more than 10% on eval, the
eraser does not erase out of sample and clause 4 is not testable: the status is then REJECTED if 1, 2 or 3 fails,
else UNCONFIRMED.

## 7. Prediction outcome (reported apart from the status)

Clauses: (c1) M1 < 20% pooled AND pooled KL <= 0.0096; (c2) pooled KL in [P/3, 3P]; (c3) R2 M1 > 50%;
(c4) F1 and F2 < 30% pooled; (c5) every base seed >= 290/298. held = all five hold; failed = any fails on data;
unsettled = a clause has no data.

## 8. Receipts

Per run: `data/runs/<run>.json` (per-step loss, KL, refit steps, in-loop probe catch on the attack split at each
refit), adapter. Per state: country-token states at readouts 10..29 fp16 (`data/ctok/<state>.npy`, Vera's rebuild),
`data/states/<state>/` (kl, lp, meta). `data/leace/` (erasers, per-readout and all-layer prices, selection table).
`proof-matrix.md` (Bruno), one row per cell.
