# Skeptic folder (Vera) — probe-switch-frozen-heads-8b

**NO SIGN-OFF YET.** No 8B training data exists at the time of writing (2026-11-18, laptop jobs 9iuyse8ape and
m7ybk186x3 queued). This folder holds my own code, validated before any 8B data, so the continuation's skeptic can
sign off from its output. I did not read or run the build's scoring code.

## Files

| file | what |
|---|---|
| `vera_check.py DATA_DIR` | rebuilds from raw rows: strict / argmax / reverse switch per model (whole graph, `whole/<run>.json` rows), Wilson 95% per model, within-pair sign test (probed A-B margin minus unprobed A-B margin), T75 from `runs/<run>.json` dev_curve, 2 x T75 budget per MLP-8 seed vs the same-seed ALL-8 (else ALL-8 s0), median RMS grad per parameter over updates 1..T75 (3x clause), cached-tail vs whole-graph gap (G1), seed-level t-interval (df n-1) clipped to [0, 1] and the min over seeds. Prints MISMATCH against every stored value. |
| `vera_readout.py OFF.npy ON.npy META.json [--pos P]` | from raw bf16 bits (uint16): G0 share per split, held-out L2 logistic (C on dev) + 2-layer MLP (width 16) probe accuracy, predicted switch max(0, 2acc-1), within-pair shuffled-label null (20 draws). No torch import (two libomp copies hang sklearn on macOS). |
| `parent17/` | copies of the 1.7B parent's raw rows (git show origin/swarm/oracle-1027-forced-attention), for validation |
| `vera_check_parent17.out` | validation: 0 mismatches. MLP-8 1.000 x3 (sign 128-0 each), ALL-8 1.000 x2, last-MLP 0.000 x2, base 0.000; RMS ratio MLP/ALL 0.59 / 0.59 / 0.58; G1 gap 0 except mlp27 s1 +0.023 |
| `vera_readout_q06_mid27.out` | validation on the build's 0.6B smoke cache: G0 1.000 all splits; last-MLP site logistic acc 0.531 (AUROC 0.548; null mean 0.506, max 0.525) vs the build's 0.551 -> predicted 0.06 vs 0.10 (the C choice moves it ~0.04) |
| `novelty.md` | prior work: Sauers watching-probe-detector (8B, all modules), Li & Kim 2603.13461, Lamparth & Reuel 2302.12461, Dong et al. 2506.01115 |

## Definition notes found while validating

- The stored `share_unprobed_B` / `share_probed_A` count the top-1 id with ties allowed; the strict share differs
  (parent mlp27 s1: 0.727 vs 0.625). Not a status input. My checker uses the parent definition and also writes the strict one.
- The frozen last-MLP prediction is method-sensitive by ~0.04 (regularisation choice). Quote it to one digit.

## For the continuation's skeptic

1. `git pull` the branch; run `python3 vera_check.py ../../data` (adjust run names if the 8B last-MLP arm is
   called mlp35; the RMS/budget block keys on prefixes all8 / mlp8).
2. For G0 and the last-MLP number, the raw caches are gitignored (`raw/`). Ask build for `data/skeptic_pack/`
   (block-35 eager post-attention at pos 79 and block-28 input at pos 71/79, probe off/on, uint16) and run
   `vera_readout.py` on it.
3. The status rule is PREREG.md's FROZEN block (polarity flipped vs 1.7B: CONFIRMED = MLP-8 strict >= 0.75 in >= 2
   of 3 seeds AND ALL-8 >= 0.75 in >= 2 seeds; an MLP-8 null counts only at >= 2 x T75 of its same-seed ALL-8).
