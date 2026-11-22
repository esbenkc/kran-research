# Skeptic folder (Vera) — probe-switch-frozen-heads-8b

**SIGNED OFF: CONFIRMED** (continuation room oracle-1122-frozen-heads-8b, 2026-11-21 PST). See `signoff.md`.
All 3 MLP-only seeds and 2 all-modules seeds rebuilt from the rows on origin (e453f7047) with `vera_check.py`
(0 mismatches, `vera_check_8b_final.out`); null, copy, adapter and audit checks in `vera_cont.py` / `vera_adapters.py`.

Reproduce (stdlib only, from this folder):

    python3 vera_check.py ../../data          # status cells, Wilson, sign test, G1, seed-level
    python3 vera_cont.py ../../data           # label-flip null, copy check, head-audit rebuild
    python3 vera_adapters.py ../../adapters   # which modules each LoRA adapter trains

State at the first room's close (2026-11-18), kept for the record: s0 cells rebuilt; J1 rows were not on origin then.

## Files

| file | what |
|---|---|
| `vera_check.py DATA_DIR` | rebuilds from raw rows: strict / argmax / reverse switch per model (whole graph, `whole/<run>.json` rows), Wilson 95% per model, within-pair sign test (probed A-B margin minus unprobed A-B margin), T75 from `runs/<run>.json` dev_curve, 2 x T75 budget per MLP-8 seed vs the same-seed ALL-8 (else ALL-8 s0), median RMS grad per parameter over updates 1..T75 (3x clause), cached-tail vs whole-graph gap (G1), seed-level t-interval (df n-1) clipped to [0, 1] and the min over seeds. Prints MISMATCH against every stored value. |
| `vera_readout.py OFF.npy ON.npy META.json [--pos P]` | from raw bf16 bits (uint16): G0 share per split, held-out L2 logistic (C on dev) + 2-layer MLP (width 16) probe accuracy, predicted switch max(0, 2acc-1), within-pair shuffled-label null (20 draws). No torch import (two libomp copies hang sklearn on macOS). |
| `vera_cont.py DATA_DIR` | continuation checks: probe-label flip null (10,000 draws), per-prompt logits sha overlap and push correlation between files, head-audit R_blind3 and null p95 rebuilt from `data/audit` (`vera_cont.out`) |
| `vera_adapters.py ADAPTER_DIR` | lists the LoRA modules in each saved adapter by walking the pickle (no torch) (`vera_adapters.out`) |
| `signoff.md` | the verdict (role / model / verdict header) |
| `parent17/` | copies of the 1.7B parent's raw rows (git show origin/swarm/oracle-1027-forced-attention), for validation |
| `vera_check_parent17.out` | validation: 0 mismatches. MLP-8 1.000 x3 (sign 128-0 each), ALL-8 1.000 x2, last-MLP 0.000 x2, base 0.000; RMS ratio MLP/ALL 0.59 / 0.59 / 0.58; G1 gap 0 except mlp27 s1 +0.023 |
| `vera_readout_q06_mid27.out` | validation on the build's 0.6B smoke cache: G0 1.000 all splits; last-MLP site logistic acc 0.531 (AUROC 0.548; null mean 0.506, max 0.525) vs the build's 0.551 -> predicted 0.06 vs 0.10 (the C choice moves it ~0.04) |
| `vera_check_q06smoke.out` | format check on the build's 0.6B chain smoke (data/whole + data/runs, 64 updates): 0 mismatches; MARK-8 strict 0.789 [0.710, 0.851], sign 127-0; every other cell 0.000 (64 updates is too short, as intended). vera_readout.py on data/skeptic_pack gives the same numbers as on raw/ (acc 0.531). |
| `novelty.md` | prior work: Sauers watching-probe-detector (8B, all modules), Li & Kim 2603.13461, Lamparth & Reuel 2302.12461, Dong et al. 2506.01115 |

## Definition notes found while validating

- The stored `share_unprobed_B` / `share_probed_A` count the top-1 id with ties allowed; the strict share differs
  (parent mlp27 s1: 0.727 vs 0.625). Not a status input. My checker uses the parent definition and also writes the strict one.
- The frozen last-MLP prediction is method-sensitive by ~0.04 (regularisation choice). Quote it to one digit.

## For the continuation's skeptic

1. `git pull` the branch; run `python3 vera_check.py ../../data` (adjust run names if the 8B last-MLP arm is
   called mlp35; the RMS/budget block keys on prefixes all8 / mlp8).
2. For G0 and the last-MLP number, the raw caches are gitignored (`raw/`). Use `data/skeptic_pack/` (J0a writes it: mid35_79.npy [704, 2, 4096] and in28_71_79.npy
   [704, 2, 2, 4096], uint16): `python3 vera_readout.py PACK PACK meta.json [--pos 0|1]`.
3. The status rule is PREREG.md's FROZEN block (polarity flipped vs 1.7B: CONFIRMED = MLP-8 strict >= 0.75 in >= 2
   of 3 seeds AND ALL-8 >= 0.75 in >= 2 seeds; an MLP-8 null counts only at >= 2 x T75 of its same-seed ALL-8).
