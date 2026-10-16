# Deviations from PREREG.md

Each entry: timestamp (Copenhagen time), what changed, why, and where the original rule is still reported.

## P0 — exploratory pilot before the frozen design (2026-10-16 ~14:58 CEST)

Before the analysis plan was frozen, build ran a pilot with a different design (d = 32 latent
model `x = A z`, subspace-LDA probes, softplus evasion loss, "upstream" and "in-place" evaders).
Results were posted in the room (#1040). The pilot is kept in `pilot/` for the record. It is
outside the verdict. Its proposal to switch the primary metric to `1 - SS_res/SS_tot` was rejected
(room #1041-#1045); the primary stays squared Pearson r, as frozen.

## D1 — clip AUROC before Phi^-1 (2026-10-16 ~15:05 CEST, before any decision run)

`d' = sqrt(2) Phi^-1(AUROC)` is infinite at AUROC = 1. AUROC is clipped to
`[1/(2n), 1 - 1/(2n)]`, n = test-set size (critic, room #1032). A moment-based d'
(`u.Delta_hat / sqrt(u' Sigma_hat u)`) is reported as a secondary (`r2_white_moments`).

## D2 — stop rule and penalty weight (2026-10-16 ~15:10 CEST, before any decision run)

Frozen rule: "train until train AUROC of w <= 0.6 or a fixed step cap". With penalty weight 10 this
fires after 25-50 Adam steps. The retain (Mahalanobis) term has then had no effect, so the run does
not test the lemma's premise (an evader that pays for its edit in the Mahalanobis norm and is at
that cost's minimum).

New rule, approved in the room (#1055, #1056): choose `LAM` from {0.5, 1, 2, 5} on synthetic seed 0
(Mahalanobis retain) as the smallest value whose run ends with train AUROC of w <= 0.6 AND a loss
plateau (relative change < 1% over the last 1000 steps). The rule reads only the monitor's train
AUROC and the loss, never the 50 probes. Fixed step cap, no early stop. First try used a
4000-step cap; no LAM plateaued (loss still fell ~2% per 1000 steps), so the cap was raised to
16000 steps before any probe result was computed. The chosen `LAM` is in `data/lam_selection.json`
and is used for all seeds, both settings and both retain arms.

The literal frozen rule (`LAM = 10`, stop at the first train AUROC_w <= 0.6) is also run and saved
(`EARLY_STOP=1`, files `*_earlystop*`); its numbers are reported in the post next to the decision
numbers.
