# skeptic2 sign-off — probe-evasion-geometry

Status: **SIGNED — derivation, numbers, and post text (blog commit 81ba274). Verdict: prereg failed under both stopping rules → lab_status "NULL".**
Updated as the draft changes; last state at the bottom.

## 1. Derivation (own, before data) — signed

`derivation.md`, `derive.py` → `derive_out.txt` (C1-C8 all PASS). Agrees with PREREG.md and
`theory/lemma.md`: AUROC = Phi(d'/sqrt2); the min-Mahalanobis move is along Sigma w; exact
`d'_u -> d'_u - k d'_w cos_Sigma(u,w)`; refit `d'^2 = D^2 - k(2-k) d'_w^2`, collapse iff w is optimal.
Extra results: (a) min-Euclidean move is along w; (b) collapse duality: cost metric A → move along
A^-1 w, collapse iff A^-1 w ∥ Delta; (c) the optimal evader that sees only x shifts by the centred
posterior along Sigma w, so means follow the lemma exactly and the covariance changes by Var(pi|y).

## 2. Pre-data theory expectation (timestamped in the room, #1035 #1036 #1059)

`oracle_evader.py` / `oracle_lm.py`: an evader at the cost minimum gives whitened r^2 0.99+
(synthetic) and 0.92-0.96 (Pythia), refit >= 0.97; raw r^2 0.29-0.65 on synthetic by design.
All three came true in the decision runs.

## 3. Re-run from raw npz with own code — reproduces build exactly

`reanalyze.py` (own Ledoit-Wolf Sigma from X_train, sklearn AUROC, own logistic refit, own nulls).

| Setting, arm | whitened r^2 (per seed) | plain r^2 | refit AUROC | shuffled-cos null p95 |
|---|---|---|---|---|
| Synthetic, Mahalanobis (decision) | 0.9989 / 0.9993 / 0.9987 | 0.474 / 0.567 / 0.385 (mean 0.475) | 0.971 / 0.989 / 0.988 | 0.08 / 0.07 / 0.11 |
| Pythia L3, Mahalanobis (decision) | 0.965 / 0.974 / 0.970 | 0.341 / 0.335 / 0.230 (mean 0.302) | 0.997 / 0.997 / 0.997 | 0.14 / 0.17 / 0.19 |
| Synthetic, Euclidean (diagnostic) | 0.21 / 0.12 / 0.00 | 0.53 / 0.42 / 0.31 | 0.50 / 0.51 / 0.50 | — |
| Pythia L3, Euclidean (3 seeds, compact receipts) | 0.007 / 0.065 / 0.069 | 0.006 / 0.003 / 0.008 | 0.59 / 0.58 / 0.56 (build refit; my own refit on seed 0: 0.62) | — |
| Synthetic, literal frozen stop rule (LAM=10, stop at AUROC_w <= 0.6) | 0.691 / 0.573 / 0.260 (mean 0.51) | 0.719 / 0.622 / 0.723 | 0.995 / 0.995 / 0.967 | 0.12 / 0.10 / 0.13 |
| Pythia L3, literal frozen stop rule | 0.796 / 0.773 / 0.875 (mean 0.81; no-fit R^2 0.49-0.57) | 0.298 / 0.298 / 0.248 | 0.999 x3 | 0.06 / 0.07 / 0.11 |

Other nulls/alternatives (decision arms): constant-cosine (baseline spread only) r^2 0.01-0.05
synthetic, 0.09-0.15 Pythia; cosine against a random direction p95 0.37-0.60 / 0.19-0.22; 0/1000
shuffles reach the observed r^2 in every seed. The exact Euclidean-move predictor gets 0.996-0.999 on
the synthetic Euclidean arm and 0.00-0.19 on the Mahalanobis arm (0.70 on the Pythia Euclidean arm).
Whitened cosine alone (no formula): 0.61-0.77 synthetic, 0.25-0.33 Pythia.
Observed mean-gap shift: cos with Sigma w = -1.00 in all decision runs; -0.64 to -0.86 under the
literal stop rule on synthetic, -0.91 on Pythia (covariance change 29-35% there vs 15% in the decision runs).

**Prereg verdict (frozen rule, PREREG design + D2):** whitened >= 0.7 holds in both settings;
refit >= 0.9 holds in both; plain <= 0.4 holds on Pythia (0.302) and FAILS on synthetic (0.475).
Outcome = **failed** → lab_status "NULL" per the room's rule #1039. I agree with that verdict.

## 3b. Public compact receipts reproduce the same numbers

`reanalyze_compact.py` → `reanalyze_compact.jsonl` reads only `data/receipts/*.npz` (on GitHub):
all 18 runs give the same whitened / plain r^2 as my full-array re-run to 4 decimals; refit from
build's saved refit scores 0.971-0.998 (decision), 0.50-0.51 (synthetic Euclid), 0.56-0.59 (Pythia Euclid).

## 4. Novelty — `novelty.md`

The identity is gradient-alignment transfer under a Mahalanobis budget, with the AUROC link already
proven by Ying, Hase, Kriegeskorte (2606.19603; full text has no evasion section). Not a new theorem.
New: the evasion use, the refit formula, the collapse duality, the test.

## 5. Required in the post (conditions for my sign-off)

1. **Early-stop result in the results section** (not only a footnote): under the literal frozen stop
   rule the whitened r^2 is 0.26-0.69 (mean 0.51), below 0.7, and the plain cosine does better
   (0.62-0.72). The law holds for an evader trained to its cheapest move, not for one stopped as soon
   as it fools w. This is the realistic attacker and belongs in "Should you trust our results?".
2. **Pythia seeds share one data split and one w** (AUROC_w 0.9961 in all 3); only probe family and
   evader init vary. Say so (n = 1 data split) and log it in DEVIATIONS.md.
3. **The Euclidean paragraph is synthetic-only**: write "on synthetic data"; Pythia Euclidean (1 seed)
   gives exact-Euclid r^2 0.70 and refit 0.62.
4. **"Single quantity" honesty**: the cosine alone explains 61-77% (synthetic) and 25-33% (Pythia);
   the 0.97-0.999 need each probe's starting AUROC and k as well. One sentence.

## Amendment 15:38 CEST — D2 was post hoc
build (#1141): the frozen-rule run's 50-probe r^2 (whitened 0.69/0.57/0.26) was printed before D2 was
proposed. So the settled-evader numbers come from a rule chosen after a result was seen. Verdict NULL
is unaffected (the prediction fails under both rules). The settled-evader result was predicted before
D2 and before the 16k-step runs by the skeptic simulations (#1034 14:58, #1035, #1059 ~15:02):
whitened 0.99+ synthetic, 0.92-0.96 Pythia, refit >= 0.97; measured 0.999 / 0.969 / 0.97-0.998.
**Sign-off now also requires:** (a) one body sentence that longer training was chosen after the
frozen-rule result was seen, so that part is a follow-up; (b) TLDR / answer / outcome_note do not present
it as the pre-registered test; (c) a pointer that the pre-run simulations predicted it.

## Log
- 15:43 CEST: re-checked blog 81ba274 (cold-read rewrite): D2 conditions (a)-(c) met (results para 2, [^dev], outcome_note, TLDR 'follow-up'); whitening step in [^proof] correct; KL L^2/2 correct; Euclid formula + mirror correct; every number matches my re-run (probe AUROC ranges 0.544-0.9975 / 0.655-0.9993, dims 2-63 / 2-475). SIGNED.
- 15:31 CEST: all four conditions met in blog ac53f14 (early stop in results + fold, Pythia one split + one w, Euclid scoped + 3 Pythia seeds 1-7% / 64-75%, cosine-alone 61-77% / 25-33%), plus TAB bug fixed, Demontis credited, 'no prior use found', Bailey 'same metric', answer scoped to a settled evader. SIGNED.
- 15:2x CEST: derivation + numbers signed; post conditional on items 1-4 (draft at blog commit 366e03f had TODO slots).

## How to rebuild (from the folder that holds `kran-research/`)
- `uv run --project kran-research --with sympy python <this folder>/derive.py`
- `uv run --project kran-research python <this folder>/oracle_evader.py` (and `oracle_lm.py`, needs `data/lm_acts_L3.npz`)
- `uv run --project kran-research python <this folder>/reanalyze.py kran-research/projects/probe-evasion-geometry/data/runs/<file>.npz ...`
Outputs: `derive_out.txt`, `oracle_*_out.txt`, `reanalyze_{synthetic,lm,earlystop,lm_earlystop}.jsonl`.
