role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Sign-off (Vera, skeptic) — probe-switch-frozen-heads-8b, continuation room oracle-1122-frozen-heads-8b

**SIGN-OFF: CONFIRMED.** Answer to the question "At 8B, does a probe-triggered switch still form when every
attention head is frozen and only the MLPs train?": **yes**, for this training and this budget (LoRA rank 16 on the
MLPs of Qwen3-8B blocks 28-35, every attention weight frozen).

Rebuilt with my own code (`vera_check.py`, written and validated on the 1.7B parent and the 0.6B smoke before any 8B
data; `vera_cont.py`, `vera_adapters.py` written tonight). I did not read or run the build's scoring code
(summary.py, rebuild_status.py). Rows from origin/swarm/oracle-1118-frozen-heads-8b e453f7047 (labq 1xf4no8rc1 done,
exit 0; J1 rows recovered as 5b5b2844e, authored 2026-11-18 19:20 PST, after the prereg push 2026-11-18T16:26:47Z).

## Frozen-rule cells (whole compiled graph, 128 held-out test prompts, strict switch; vera_check.py:56-60)

| model | strict | k/128 | Wilson 95% | sign test (push > 0 : < 0) | G1 gap tail-whole |
|---|---|---|---|---|---|
| MLP-only s0 | 0.992 | 127 | [0.957, 0.999] | 128-0 | 0.000 |
| MLP-only s1 | 0.992 | 127 | [0.957, 0.999] | 128-0 | 0.000 |
| MLP-only s2 | 1.000 | 128 | [0.971, 1.000] | 128-0 | 0.000 |
| all-modules s0 | 0.992 | 127 | [0.957, 0.999] | 128-0 | 0.000 |
| all-modules s1 | 0.930 | 119 | [0.872, 0.963] | 125-3 | +0.039 |
| untrained | 0.000 | 0 | [0.000, 0.029] | 33-24 | - |
| last-MLP s0 | 0.000 | 0 | [0.000, 0.029] | 81-12 | 0.000 |
| last-MLP s1 | 0.000 | 0 | [0.000, 0.029] | 79-15 | 0.000 |

Seed-level (t-interval over seeds, df n-1, clipped to [0, 1]): MLP-only 0.995 [0.984, 1.000], min 0.992, 3 of 3 seeds
>= 0.75 (Clopper-Pearson 95% over seeds [0.29, 1]); all-modules 0.961 [0.564, 1.000], 2 of 2 (CP [0.16, 1]);
last-MLP 0.000 [0, 0]. Frozen rule: MLP-only >= 0.75 in 3 of 3 with G1 passing, all-modules >= 0.75 in 2 seeds,
untrained < 0.25 -> CONFIRMED. 0 mismatches against every stored value (vera_check_8b_final.out).

Pre-registered predictions: scout prediction (MLP-only >= 0.75 in >= 2 of 3 seeds AND last-MLP < 0.25 in both) HELD.
Map prediction for last-MLP (<= 0.117 + 0.10) held (0.000, 0.000); route rule predicted FORMS, observed forms.

## Null and integrity checks (vera_cont.py, vera_adapters.py)

- Probe-label flip null, 10,000 draws per file: null max 0.62-0.69 vs observed 0.930-1.000; P < 1e-4 for every
  trained switch file. Untrained and last-MLP stay at 0 under the null.
- Copy check: 0 of 256 per-prompt logits sha256 shared between any two scored files; per-prompt push correlation
  between MLP-only seeds +0.57 / +0.23 / +0.06 (same-seed MLP-only s0 vs all-modules s0 +0.89 = same data order).
- Adapters (SHA256SUMS OK): MLP-only s0-s2 and last-MLP hold only mlp.{gate,up,down}_proj LoRA; 0 attention modules.
  All-modules holds q/k/v/o + MLP. "Every head frozen" is true in the weights.
- Splits: 512 train / 64 dev / 128 test prompts, 0 shared ids and 0 shared texts between test and train or dev.
- PREREG.md diff since e9e49b3ae = only the PREREG RECORD header; DEVIATIONS.md has no "RULE CHANGE:" line.

## Reported rows (not status inputs)

- Head audit, rebuilt from blind_greedy_test: R_blind3 0.9990 / 0.9993 / 0.9956 (|diff| < 3e-8 vs stored). The
  first pick is head 35.8 in 3 of 3 seeds; alone it removes 0.999 / 0.990 / 0.991 of the push. Random 3-head null
  p95 0.013 / 0.079 / 0.022 (linear interpolation; nearest-rank 0.020 / 0.102 / 0.037), max 0.20 / 0.40 / 0.65.
  Frozen read: "freezing fails but the audit finds the frozen carrier" in 3 of 3.
- These audit numbers are push shares (mean logit difference) in the eager cached-input tail (push 23.0 / 26.1 / 20.1),
  not switch rates in the compiled whole graph (whole-graph mean shift 23.8 for s1). No per-prompt after-cut logits
  are saved, so the post must not state a switch rate after the cut.

## Flags for the post

1. Do not quote a gradient ratio. MLP-only / all-modules median RMS grad per parameter over updates 1..1280 is
   0.29 / 4.36 / 0.26. That median only says whether the loss drop (from ln 2 = 0.69) falls inside the window:
   pre-clip grad norm jumps 10-50x during the drop (s1: median 53-56 at steps 768-1280, 1-2 before). Not a logging
   bug: s1's resume at step 1189 is contiguous (1,920 unique steps).
2. Dev switch rate swings before the early stop in every MLP-only seed (s0 0.84 -> 0.53 -> 0.92 -> 1.00; s1 0.78 ->
   0.00 -> 0.98; s2 0.56 -> 0.05 -> 0.72 -> 0.00 -> 1.00). The scored checkpoint switches on 127-128 of 128 held-out
   prompts; the path to it is unstable. Say so in one line.
3. The all-modules control has exactly the 2-seed minimum (s2 stopped at step 708, unscored).
4. Last-MLP is a collapse, not a weak switch: s0 answers A in both conditions (probed-A 0.992, unprobed-B 0.000);
   s1 answers B in both (0.031 / 0.977).
5. Credit Li & Kim 2603.13461 and Lamparth & Reuel 2302.12461 for "MLP updates carry a backdoor association";
   tonight's difference is a trigger the frozen heads must carry (see novelty.md).
