# Hugo (skeptic2) sign-off — theory-1019-max-pool-survives

Verdict: **SIGN-OFF: REJECTED reproduces** (the answer to the title is No). Prereg outcome: failed.

Note: Hugo ran on Claude opus from 15:00 (the openai/gpt-5.5 session never reached the room).

## What I rebuilt, from raw data only
`rebuild.py` reads `data/split.json`, `data/probe.npz`, `data/clean_tokens.npz` and
`data/runs/lam0__seed{0,1,2}__tokens.npz`. It does not import or read the build code. It fits its own
tokenwise logistic probe (sklearn, standardised features, C by 5-fold GroupKFold on city, on max-pooled
AUROC), pools, re-thresholds (2nd-highest of 200 refit TRUE on the attacked model) and bootstraps
(city clusters, 2000 draws, rng 0, threshold re-picked per draw). Output: `rebuild.log`, `rebuild_out.json`.
Bruno's `scores.npz` is loaded only at the end for a correlation check.

| monitor | clean catch @1% | attacked catch per seed | pooled | 95% CI | attacked AUROC |
|---|---|---|---|---|---|
| frozen (mean L14) | 96.0% | 0.67 / 0.84 / 0.84% | 0.78% | 0–5.8% | 0.583 / 0.573 / 0.571 |
| **max14** | **91.4%** | **0 / 0 / 0%** | **0.0%** | **0–0.89%** | 0.523 / 0.501 / 0.504 |
| last14 (control) | 99.7% | 1.17 / 0.67 / 0.50% | 0.78% | 0–14.5% | 0.615 / 0.652 / 0.714 |

- My C pick = 0.01 for max14 (cv AUROC 0.992), same as the build. Correlation of my scores with
  `results/scores.npz`: max14 0.999999, frozen/last14 1.000000 per seed.
- Repro gate: frozen AUROC 0.583/0.573/0.571 vs 1018 0.583/0.573/0.571 (met).
- Token files are the attacked model: frozen-probe mean over `lam0__seed*__tokens.npz` equals the attack-time
  step-150 scores in `lam0__seed*.npz` to max |diff| 0.0000 (`consistency.log`).
- Status rule: upper bound 0.89% < 50% -> REJECTED. Prereg part 1 decisively missed.
- Framing rule: max14 − last14 paired CI [−14.2, +0.34] pts: lower bound < 0, so no "pooling helps" sentence.

## Robustness of the REJECT (my additions)
- Every C, not only the CV pick: C in {0.001, 0.01, 0.1, 1, 10} gives attacked catch 0/0/0% for every C
  (clean 94.3 / 91.4 / 33.1 / 9.9 / 7.0%).
- Held-out-half threshold (t from 50 refit cities' true statements): 0.67/0.34/1.0% and 0/0/0%.
- Looser alarm rates: 5% FPR -> 1.0/1.7/1.0%; 10% FPR -> 7.4/6.4/7.4% (about the alarm rate: no signal).
- Max over label-bearing tokens only (tokens after the first token that differs from the twin statement;
  22.9% of pooled tokens): clean 99.7% -> attacked 1.0/0.5/0.3%. So the 0% is not the prefix hiding a
  surviving answer-token signal; the answer tokens lost the signal along the clean direction.
- Separation on label-bearing tokens (eval split, max14 direction, false minus true, in units of the clean
  token SD): 1.84 clean -> 0.25/0.26/0.28 attacked. False answer tokens fall below the prefix mean
  (+0.89 -> −0.10/−0.19/−0.20 SD). Argmax on a shared-prefix token for eval false: 0.7% clean ->
  85.6/95.5/96.6% attacked (build: 86/95/97%).

## Nulls
- Random-direction max probe (same norm as max14): clean catch 1.7%, attacked 1.2/0.7/0.7%, AUROC ~0.50.
- Within-pair label-swap max14 (my own swap, rng 0): clean 2.3%, attacked 1.3/0.7/0.7%.
- Both sit at the 1% alarm rate. The real max14 is at 91.4% clean, so the clean ceiling is not luck;
  the attacked max14 (0%) is at or below the null level: indistinguishable from no monitor.

## Derivation (my own) and checks
1. Prefix lemma. A causal decoder computes h_t from tokens 1..t only. A true/false pair of one city and
   template shares tokens 1..k, so h_t is the same for t <= k under ANY weights (clean or LoRA). Every probe
   score on the prefix is then equal inside the pair: per-token separation is 0 there. Measured: max prefix
   |diff| / median token norm at L14 = 6.6e-3 clean, 4.1e-3 / 5.2e-3 / 6.3e-3 attacked (< 1e-2 bar; bf16
   batching noise).
2. Consequence for max pooling (mine, used to read the result): max_t s_t(F) >= max_{t<=k} s_t = the same
   prefix floor as the twin TRUE statement. When the attack pushes answer tokens under that floor, the twin
   scores tie, and paired AUROC goes to 0.5. That is what the data show (argmax on prefix 86–97%).
3. Shared shift. max_t(s_t − a) = max_t s_t − a and mean_t(s_t − a) = mean_t s_t − a; the 2nd-highest true
   score moves by −a too, so the catch is unchanged for both poolings. `derivation_check.py`: catch 0.796
   (max) and 0.482 (mean) identical before/after a = 5.3.
4. Bell-curve formula Phi(D − (1−c)p − z), z = Phi^-1(1 − 2/201). Re-derived: t1 = mu_T1 + z sigma,
   mu_F1 − mu_T1 = sigma(D − p + cp). MC of the finite 2nd-of-200 rule: 0.535 vs formula 0.568 at
   D=4, c=0.5, p=3 (Iris: 0.534), so the formula overstates by ~0.02–0.04; disclosed as D3.

## Not recomputed (disclosed)
- Output change of the attack (held-out KL 0.0070/0.0068/0.0072 nats/token) is read from the run JSONs;
  recomputing it needs the 1.5B model forward pass. I did not recompute it.
- The fresh probe (36.1% vs 12.6% by CV split) is outside the status; I did not rebuild it.
