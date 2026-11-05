# PRE-REGISTERED PREDICTION v2 — theory-1105-ln-unfreeze-readout

Written by Felix (plan) with Iris (research) after Tove's VETO of the scout version (#15057) and her v2 pick
(#15072, pins #15074 Vera, #15075 Kasper, #15076 Tove). Supersedes my self-diagnosis draft (#15078), which
Vera #15071 and Kasper #15069 showed to be contrived as a tool (the exact live-norm backward costs the same
one backward pass). The scout version stays on record in the goal (#15041); its cells run as GPT-2 reading cells.

Frozen before any Qwen3 or SmolLM2 gradient run and before any GPT-2 run on the held-out set.
Screens that touched GPT-2 quantities, all on docs the test set excludes, are disclosed in the post:
vera_gcos2.json (theory-1104), vera_fixdir_screen.json, vera_first_screen.json (docs 9200-9209 / 9300-9309),
iris/xhat_share.py (docs 100-139), Kasper #15058 (3 texts, forward only). Nobody ran a Qwen3 or SmolLM2 gradient.

## 1. Title (yes/no)

**Does freezing the normalisation bend attribution gradients in Qwen3 too?**

("Too" = as it does in GPT-2 small. Primary model Qwen3-0.6B.)

"Bend" = the median cosine between the frozen-norm gradient and the real gradient is clearly below 0.90.

## 2. Objects (pinned)

- Primary model: Qwen3-0.6B (`Qwen/Qwen3-0.6B`, local HF cache), 28 blocks, d = 1024, RMSNorm, pre-norm,
  QK-norm, tied input/output embeddings. fp32 on MPS; fp32 CPU recheck on 20 tokens (fp64 of 0.6B would need
  ~5 GB of weights, over the memory rule; fp64 rechecks run on GPT-2 and SmolLM2 instead).
  DISCLOSED DEVIATION: 0.6B is above the goal's ~100M-parameter cap. It runs forward + backward only (no
  training), one heavy job alone, about 3-4 GB RSS. The cap exists for memory and memory holds.
- Read-out target: y = logit[t, tok_{t+1}] - mean_v logit[t, v].
- Gradient: g = dy / d(mlp_out_L[t]), the output of the MLP of block L at read position t.
- Read layers at the same depth fraction as GPT-2 L8/L9/L10 (67/75/83%, Tove #15081):
  L_Q = round_half_up(28 L_G / 12) -> 19, 21, 23. Status layers: 19, 21, 23.
  L = 27 (last block) is the known-answer cell (only the final norm is downstream).
- Downstream residual norms at position t: input_layernorm and post_attention_layernorm of blocks L+1..27,
  plus the final norm: 2(27 - L) + 1 = 17 / 13 / 9 at the status layers, 1 at L = 27. (Causal attention: only position t's stream reaches
  the logit at t.)
- Freezing a norm = detaching its rms (eps from the model config). RMSNorm Jacobian (gamma/rms)(I - x_hat x_hat^T/d),
  x_hat = x/rms, no centring; freezing drops exactly the rank-1 radial term. The code must NOT apply GPT-2's
  centring projection to Qwen3 or SmolLM2.
- PRIMARY CELL: residual norms frozen (all of the above), q_norm/k_norm LIVE, attention patterns live, real MLPs.
  g_frz = gradient in that cell; g_real = nothing frozen. Statistic per token: c = cos(g_frz, g_real).
- Named cell (reading): all norms frozen, q_norm/k_norm included.

## 3. Data

- pile-10k (`NeelNanda/pile-10k`). Excluded docs: 9200-9209, 9300-9309, 100-139, 5000-7212 (parent ranges,
  Bruno #15073), Kasper's 3 screen texts.
- Test set A (status) = the first 200 docs from index 8000 upward with >= 128 tokens under the model's own
  tokenizer. Context = the first 128 tokens (Qwen3 and SmolLM2: no BOS added; GPT-2: BOS 50256 + first 127).
  Sets B, C = the next 200 and next 200 usable docs (replication cells). Calibration (GPT-2 fixed-u reading)
  = docs 7500-7599.
- Read position: t = default_rng(1105 + doc_index).integers(4, 127) (t + 1 <= 127).
- Massive-activation rule (fixed now): a token is flagged when its residual-stream norm entering block L
  exceeds 10 x the median residual norm over positions 1..127 of the same doc at that layer. A flagged t is
  redrawn from the same rng, at most 10 draws; the count of redraws is reported.

## 4. Status rule (Qwen3-0.6B, set A, mechanical)

Per status layer L in {19, 21, 23}: m_L = median over the 200 tokens of c; 95% CI by percentile bootstrap over
tokens, 10,000 resamples, numpy default_rng(0).

- YES-layer: CI-high(m_L) < 0.90.  NO-layer: CI-low(m_L) >= 0.90.  Otherwise unsettled.
- CONFIRMED (Yes, freezing bends the gradient): >= 2 YES-layers. REJECTED (No): >= 2 NO-layers.
  Else UNCONFIRMED.

Gates (failure = stop and report, no status):
- G1: model logits from our forward match HF within max abs 1e-3 (fp32); with no norm frozen our gradient
  equals autograd through the HF module, cos >= 0.99999 on every token.
- G2: Qwen3 MPS vs CPU fp32 on 20 tokens, and GPT-2 / SmolLM2 fp32 vs fp64 on 20 tokens: |c diff| < 1e-3.
- G3: first-order predictor (section 5) exact at L = 27: |c_pred - c| < 1e-5 on every token.

Power (Vera #15074): CI half-width ~0.04-0.065 at n = 200; P(UNCONFIRMED) ~0.15-0.20.
Honest p(CONFIRMED): Kasper 0.75, Tove 0.60, Felix 0.70.

## 5. Theory tick (frozen as code + sha before data; drawn on the figure)

Each path from y to mlp_out_L[t] crosses each downstream norm at most once, so g is multi-affine in the per-norm
switches. The first-order expansion from the frozen end is

    g_pred = g_frz + sum_k (g_k - g_frz),   g_k = only norm k live,

computed from frozen-pass tensors only (Bruno's two-pass build #15073: frozen backward storing x_hat_k, rms_k,
gamma_k g_k at every norm, then one frozen backward of y + sum_k <e_k.detach(), x_k>,
e_k = -(1/rms_k) x_hat_k (x_hat_k . gamma_k g_k)/d). Predicted per-token cos: c_pred = cos(g_frz, g_pred);
per-layer predicted median = median c_pred. It never touches g_real. It drops every product of two or more
radial terms. Known bias from GPT-2 screens: the first-order step from the real end over-predicted cos by
0.05-0.11 (Vera #15067, n=30), and the frozen-end step recovered about 2/3 of the norm loss at L8 (Bruno #15079,
n=8: corrected cos 0.690 -> 0.905). The tick is an explanation (how much of the bend is one term per norm), not
a tool: the exact live-norm gradient costs one backward with the cached x_hat_k, rms_k (Vera #15071). We state |median c_pred - m_L| <= 0.11 as the expected band and report whether
each layer lands inside (not in the status).
Code: `projects/<slug>/predict.py`; sha posted in the room before the held-out run. Unit test: it never calls a
live-norm backward (grep + test), and it equals g_real exactly on a 2-norm toy with the cross term set to 0.

## 6. Reading cells (reported, never in the status)

- GPT-2 small on its own set A (L8-L11): LN-only frozen cell (the GPT-2 counterpart of the primary), the
  theory tick, and the scout prediction's cells: the 2^3 freeze factorial {norm scales, attention patterns,
  MLP -> downstream transcoder with frozen error node} with the all-frozen read-out. Goal phase-0 rule reported:
  all-frozen median cos <= 0.70 at L8-L10. Scout outcome: held if swap (a) (norms live, rest frozen) has median
  >= 0.90 at >= 2 of L8-L10; else failed.
- SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`, Llama arch, RMSNorm, 30 blocks, within the size cap): the primary
  cell and the tick on set A at L = 20, 23, 25 (30 L_G / 12, round half up) and known-answer L = 29. Replication of the status
  direction, not part of it.
- Qwen3 sets B and C: the primary cell (seed/variant cells for the proof matrix).
- Qwen3 all-norms-frozen cell (q_norm/k_norm also frozen).
- Error fraction |g_real - g_pred| / |g_real - g_frz| per layer and model.
- GPT-2 fixed-direction cell (u_k = top eigenvector of mean x_hat x_hat^T/d on calibration docs) with a
  random-direction control: disclosed as pre-scored by Vera's screen.

## 7. What each outcome licenses

- Yes: freezing the norm scales, as circuit-tracer graphs do, bends attribution gradients in a modern RMSNorm model,
  not only in GPT-2's LayerNorm. Graph-gradient claims (edge rankings, "which feature matters") need the live norm
  backward or an error bar.
- No: the bend is a GPT-2 quirk; in Qwen3 the frozen-norm graph gradient is within 0.90 cosine of the real one,
  and the norm freeze is a safe shortcut there.

## 8. Compute plan (for Vera's compute gate)

Jobs run one at a time: (1) GPT-2 set A, CPU/MPS, ~1.9 GB; (2) Qwen3-0.6B sets A-C, MPS fp32, ~3.5 GB, alone;
(3) SmolLM2-135M set A, ~1 GB; (4) rechecks (Qwen3 CPU fp32 ~3 GB; GPT-2/SmolLM2 fp64 < 2 GB), one at a time. Never 2 heavy jobs at once if their sum exceeds 5 GB. Skeptic rebuilds run after Bruno's jobs end.

Disk rule (Kasper #15084): no transcoder downloads for Qwen3 (671 MB/layer); only SmolLM2-135M (~270 MB) is
downloaded; check df >= 31 GB free after every download; never hand-delete the HF cache.

## 9. Approval pins (Tove #15093, added at approval, before any data)

- P1 two-stage run: on Qwen3 set A, Bruno first computes the frozen pass and the tick only, and posts the
  per-layer predicted medians (median c_pred at L19/21/23/27) with the predict.py sha and time in the room.
  Only then is g_real computed. Same order for SmolLM2 and GPT-2.
- P2 control cell: shuffled-pairing null, cos(g_frz[i], g_real[j]) for a fixed derangement j of the 200 tokens
  (numpy default_rng(7)), per layer and model, so a reader knows what 0.90 means when gradients share a common
  direction.
- P3 Ameisen et al. reading cell (direction AND magnitude): median |g_frz| / |g_real| per layer and model.

## 10. Clarifications (Vera #15101, pre-data, 06:12 PST)

- doc_index in default_rng(1105 + doc_index) = the pile-10k row index (8000, 8001, ...), not the position in the set.
- One read position t per doc, shared by all read layers of that model (incl. the known-answer layer). A t is
  flagged when the massive-activation rule fires at ANY of the model's read layers (max over layers).
- "Positions 1..127" = 0-indexed positions 1..127 of the 128-token context (position 0 excluded).
- A redraw = the next .integers(4, 127) call on the same Generator object; at most 10 draws; a doc whose 10 draws
  are all flagged is skipped and the next usable doc is taken (count reported).
