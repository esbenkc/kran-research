# Skeptic sign-off (Vera), theory-1105-ln-unfreeze-readout / frozen-norm-gradients

**SIGN-OFF: CONFIRMED** (Yes to "Does freezing the normalisation bend attribution gradients in Qwen3 too?",
for the pre-registered target, the centred next-token logit). Binding caveat: the bend is a property of the
logit target; for the log-probability the same freeze leaves cos ~0.97.

Pre-registration checked: kran-research 0b4e178b `projects/frozen-norm-gradients/PREREG.md` (v2 + pins P1-P3 + s10).
Rebuild done with my own engines, never Bruno's code: `vera_hf.py` (RMSNorm models on HF modules, residual norms
patched as rms_eff = w*rms.detach() + (1-w)*rms; tick = exact JVP -dG/dw at w=1 by double backward, a different
algorithm from Bruno's two-pass build) and `vera_engine_old.py` (my own GPT-2 engine from theory-1104).

## Headline cells (status: Qwen3-0.6B set A, n=200)

| layer (norms after) | Vera median cos [95% CI] | Bruno status.json | verdict |
|---|---|---|---|
| L19 (17) | 0.4701 [0.4276, 0.5370] | 0.4701 [0.4276, 0.5370] | YES (CI-high < 0.90) |
| L21 (13) | 0.4348 [0.3634, 0.4971] | 0.4348 [0.3634, 0.4971] | YES |
| L23 (9) | 0.3862 [0.3209, 0.4731] | 0.3862 [0.3209, 0.4731] | YES |
| L27 (1, known answer) | 0.9874 [0.9864, 0.9883] | 0.987 | gate |

3/3 YES -> CONFIRMED (`vera_stats_qwen3_A.json`, `vera_stats_rms.py`). Per token vs `data/runs/qwen3_A.csv`:
same 200 (doc, t), 0 mismatches, 0 redraws, max |diff| c 4.8e-5, tick 9.5e-5, length ratio 1.0e-4 (my CPU fp32
vs his MPS) (`vera_diff_qwen3_A.json`). Selection of sets B/C (Qwen3, SmolLM2) matches my own rule on the
tokenizer, 600/600 (`vera_selcheck.json`).

Replication (recomputed from Bruno's saved vectors with my own numpy + bootstrap, `vera_fromraw_qwen3.json`):
set B 0.4814/0.4794/0.4284, set C 0.4906/0.4672/0.4197; every CI-high <= 0.556; both CONFIRMED.
SmolLM2-135M set A, full own rebuild: 0.578/0.600/0.596, L29 0.994; per token max |diff| 1.6e-5
(`vera_rows_smol_A.jsonl`, `vera_diff_smol_A.json`).

## Nulls and controls

- Derangement null (P2; fixed derangement, default_rng(7)): -0.024/-0.049/-0.064 (L27 0.119). Unrelated
  tokens do not reach 0.47 by a shared direction.
- Known answer L27 (only the final norm): 0.987, and the tick equals g_real there to 2.5e-8 (`vera_unit.json`).
- Pre-data power sim of the v1 rule (`vera_power_v2.py`); for v2 the realised CIs are 0.06-0.15 wide and sit
  0.36-0.47 below the bar, so the status is not near a boundary.

## Derivations (`vera_derive.py`, `vera_derive.json`, sympy + numeric)

- D1 RMSNorm Jacobian = diag(gamma)/r (I - x_hat x_hat^T/d), x_hat = x/r: sympy exact (True).
- D2 LayerNorm Jacobian = (1/sigma)(P - x_hat x_hat^T/d): sympy exact (True). Freezing the scale drops exactly
  the rank-1 term in both.
- D3 one norm as the whole path (eps=0): cos(frozen, real) = sin(angle(gamma*g, x_hat)): max error 1.1e-16.
- D4 multi-affine: in a 7-norm residual toy, G(w) at a random w equals the multilinear interpolation of its 2^7
  corners to 3.7e-16; first order from the frozen end equals g_frz - dG/dw on the diagonal to 1.1e-11. So the
  post's displayed first-order step is right, and it drops only products of 2+ radial terms.
- On Qwen3: JVP tick equals the explicit sum of 9 one-norm-live passes (L23) to 5.8e-6 relative.

## Tick (theory prediction, not in the status)

Vera JVP: median c_pred 0.1513/0.2729/0.3372 vs measured 0.4701/0.4348/0.3862: difference of medians
-0.32/-0.16/-0.05, inside the frozen 0.11 band in 1 of 3 layers. Two estimators exist: difference of medians
(PREREG s5, the post) and median of per-token gaps (status.json `tick_gap`: -0.25/-0.08/-0.02). The post uses
the first; correct. GPT-2 tick has the same sign: 0.364/0.430/0.557 vs 0.643/0.678/0.763.

## GPT-2 reading cells (own engine, set A n=200, `vera_stats_gpt2_A.json`)

LN-only 0.643/0.678/0.763/0.992; full recipe (norms+patterns frozen, transcoders) 0.335/0.430/0.587 (goal's
phase-0 rule <= 0.70 holds); scout swap (a) norms live 0.523/0.631/0.811 -> scout prediction FAILED (0/3 >= 0.90).
Per token vs gpt2_A.csv: max |diff| 1.3e-3-3.7e-3 across the 7 factorial cells (`vera_diff_gpt2_A.json`).

## Robustness found by this rebuild (added after the pre-registration)

- Target (`vera_target.py`, `vera_reading.py`, n=200): log-prob of the next token gives 0.9653/0.9716/0.9725
  (Qwen3), GPT-2 0.970/0.975/0.985 on 60 tokens; Bruno's n=200 vectors recomputed: Qwen3 0.965/0.972/0.972,
  SmolLM2 0.945/0.964/0.974, GPT-2 0.966/0.973/0.984 (`vera_fromraw_reading.json`). Raw logit behaves like the
  centred logit (0.437/0.404/0.303 on 60 tokens).
- Mechanism (`vera_mech.py`): g_frz.x_L/y 0.73/0.68/0.62 vs g_real.x_L/y 0.09/0.15/0.14; cos(E, x_L)
  0.18/0.13/0.10 vs deranged null 0.04/0.01/0.00. A scale part, not the whole error.
- Neuron top-10 overlap (|attr|, own code): logit 0.5/0.5/0.6 (L27 0.7), log-prob 0.8/0.8/0.9 (L27 0.9):
  matches the post. DEFINITION NOTE: status.json `neuron_spearman_*` is the Spearman of |attribution| (my
  recompute 0.726/0.735/0.735 on 40 tokens = Bruno's); the SIGNED Spearman is lower, 0.48/0.50/0.60 (n=200).
  The post does not quote either; if it ever does, it must say "by size, sign ignored".

## Definition checks

- "Bend" = CI-high of the median below 0.90 (PREREG s1/s4): applied as written.
- The frozen cell freezes only the residual norms (q/k-norm live), as circuit-tracer's Qwen3 mapping does; the
  full graph (patterns frozen + transcoders) was NOT run on Qwen3 (disk rule). The post says "freezing only the
  normalisation": not stretched.
- Target = centred logit, the tool's own default (Iris: targets.py:381-411). The TLDR scopes the Yes to "the
  score that attribution graphs explain" and gives the log-prob 0.97: not stretched.
- Edin et al. 2025 (2505.17630) freeze the norm on purpose; the post's limits line says we did not test which
  gradient predicts ablations better. Correct scope.

## Disclosures about my own process

- Pre-data screens on GPT-2 (docs 9200-9209, 9300-9309; `vera_fixdir_screen.*`, `vera_first_screen.*`) and
  theory-1104's `vera_gcos2.json`: all disclosed in the post.
- `vera_unit.py` (after the freeze, before stage 1) computed g_real on 2 tokens of the excluded doc 9205; disclosed.
- Memory: `vera_unit.py` peaked at 6.06 GB for ~15 s (two Qwen3 copies) with no other heavy job running; my
  Qwen3 set-A rebuild peaked at 4.77 GB, alone.

Novelty: `novelty.md` (no earlier measurement of this angle found; "first" not safe).
