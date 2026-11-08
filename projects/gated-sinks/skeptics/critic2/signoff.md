# critic2 sign-off: gated-sinks (theory-1108-gated-sinks)

Reviewer: Kasper (critic2). The first pass ran on openai/gpt-5.5 (room #17474, PASS from the results jsons; it
capped at $2.59 before it saved receipts). This folder is the Claude (opus) restart: an own rebuild from the raw
acts_*.npz, with no import of measure.py, aggregate.py or any teammate's code.

Verdict: PASS. Status UNCONFIRMED is what PREREG section 3 gives; the registered prediction (section 0) FAILED.

Rebuild (`python3 rebuild.py > rebuild.out`, 2.3 s, numpy + scipy only):
- 24 checkpoints (12 counted at 12k + 12 matched-loss), recovery from the saved SAE weights and H_scored:
  max |rebuild - json| = 1.9e-05 (recovery), 1.7e-04 (sink mass, mean of A0_sink_block).
- Counted grid: line on 9 non-gated models a 0.2768, b +0.2487, 95% CI [-0.1209, +0.6182], s_tot 0.0852,
  R 0.584 (D4 guard not triggered). Gated residuals -0.0467 / -0.0495 / -0.0341, band 0.1705: 3/3 in band.
  Swing upper end 0.361 >= 0.05, so REJECTED(i) does not fire; (ii) and (iii) do not fire; CI covers 0, so
  not CONFIRMED -> UNCONFIRMED. Matches post TLDR (+0.25, -0.12 to +0.62) and 'band 0.17 wide'.
- Equal-training pairs (gated - standard, same seed): -0.1811 / +0.0134 / -0.0096 = post '-0.18, +0.01, -0.01'.
- Loss as a second predictor: sink slope +0.1446, CI [-0.1596, +0.4488] = post '+0.14 (-0.16 to +0.45)'.
- Registered row at matched loss (pair loss gaps <= 0.04%): gains -0.1546 / -0.0390 / -0.0280 (needs >= +0.10),
  sink ratios 0.232 / 0.943 / 0.906 (needs < 0.2), norm R^2 max over latents 0.0227 (< 0.2, holds). FAILED on
  two of three clauses. Matches prereg outcome_note '0.15, 0.04 and 0.03 lower' and '0.23 to 0.94'.
- Within-standard and same-layer cells: sink model 0.4177 vs four sink-free standard runs 0.2285-0.2626
  (post '0.42 against 0.23 to 0.26'); softmax plus one 0.4305, sink slot 0.4161 (post '0.43 and 0.42').
- Hiding check: per-token norm ridge R^2 (mean over 64 causes) <= 0.00095 in all 24 checkpoints
  (post 'under 0.2%'); gate R^2 0.059-0.065 vs gated readout R^2 0.113-0.128 (post '0.06 against 0.11 to 0.13').

Preview: https://deploy-preview-179--kranai.netlify.app/gated-sinks/ (built from 644c1c5; 0bce48d changes only
lab_cost_usd 0.3 -> 3.35 = room API spend 3.05 incl. critic2 2.59 + scout 0.30). TLDR opens 'Not settled.',
no single-$ math, no stray table pipes in paragraphs. PR #179 stays draft: UNCONFIRMED does not merge.

Note: numpy on macOS arm64 prints spurious 'divide by zero / overflow in matmul' RuntimeWarnings for these
float32 products; every output is finite and matches the jsons to 2e-05.
