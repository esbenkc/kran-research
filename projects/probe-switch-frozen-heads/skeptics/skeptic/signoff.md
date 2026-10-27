# Sign-off — Vera (skeptic), room oracle-1027-forced-attention

Title: "Does a probe-triggered switch need a trainable attention head?" (Yes = MLP-only training with every head of
the trained blocks frozen does not learn the switch; No = it does.)
Stack: Qwen3-1.7B bf16, Apple M4, torch 2.14.1; probe = leaking hook on block-14 `self_attn.q_norm` (compiled graph);
LoRA r16 a32 from scratch on blocks 20-27; PREREG v1-v1.4 (0491fcb ... 104b1f5).

I did not read build's or research's code (train.py, tail.py, audit.py, eval_whole.py, map_states.py, map_probe.py).
Every number below comes from build's saved raw files (pass-A block-20 cache, eager map states, adapters, per-prompt
rows, grad logs) run through my own scripts in this folder:
- `vera_switch.py`: switch cells from per-prompt rows (strict = top-1 strictly above top-2, PREREG), Wilson CI,
  reverse switch, ties, shuffled-condition null (2000 per-prompt swaps).
- `vera_tail.py`: own CPU fp32 Qwen3 tail (blocks 20-27 + final norm + tied lm_head), own layer math, weights streamed
  per layer from the HF safetensors, LoRA merged W + 2.0 * B @ A from the adapter file only.
- `vera_clamp.py`: same tail with the attention output at the answer position replaced by its unprobed value, per block.
- `vera_pos.py`: same tail with the MLP LoRA delta kept only at the answer position / only at positions 0-78.
- `vera_map.py`: answer-token readability map (logistic, C by pair-grouped CV; 2-layer MLP probe width 64, 3 seeds;
  mean-difference direction; within-pair condition-flip null).

## Verdict on the title

**REJECTED (No), confirmed by my rebuild.** PREREG status rule: REJECTED = MLP-8 strict switch >= 0.75 in >= 2 seeds
with G1 passing. MLP-8 strict switch = 1.000 (128/128, Wilson 0.971-1.000) in all 3 seeds on build's whole-graph
rows (title score), 1.000 in all 3 seeds on build's cached tail (G1 gap 0.00), and 1.000 in all 3 seeds in my own CPU
fp32 tail that merges only the adapter's 48 MLP tensors. Prediction outcome: FAILED (clause 2: MLP-8 < 25% fails in
every seed). Clauses 1, 3, 4 hold (see table). Scope: this LoRA (r16, lr 1e-4, <= 2048 updates), blocks 20-27, this
leaking build, Qwen3-1.7B on an M4. The switch still runs through attention heads (frozen ones): the "No" is about
TRAINABLE heads, not about attention.

## Key numbers I rebuilt (mine vs build)

| Cell | mine | build | my artifact |
|---|---|---|---|
| Clause 1: max answer-token AUROC, block inputs 20-27 (eager, base) | 0.538 (in22) | 0.540 (in22) | out_map_eager79.json |
| MLP-27 prediction 2acc-1 at mid27@79 (logistic) | 0.055 (acc 0.527) | 0.070 (acc 0.535) | out_map_eager79.json |
| <|im_end|> (pos 71) AUROC, in20..mid26 | 0.78-0.985; paired test 1.000 everywhere | 0.90 (in26, prior room) | out_map_eager71.json |
| ALL-8 s0 strict, whole rows / my CPU tail | 1.000 / 1.000 (push 30.06) | 1.000 (push 25.05 whole, 31.53 tail) | vera_switch on data/whole; out/tail_all8_s0.json |
| ALL-8 s1 strict, whole rows | 1.000 (push 34.58) | 1.000 | vera_switch on data/whole/all8_none_s1.json |
| MLP-8 s0 strict, whole rows / my CPU tail | 1.000 / 1.000 (push 24.27) | 1.000 (22.41 whole, 24.33 tail) | out/tail_mlp8_s0.json, out/pos_mlp8_s0.json |
| MLP-8 s1 strict, whole rows / my CPU tail | 1.000 / 1.000 (push 20.25) | 1.000 (20.09 whole, 20.57 tail) | out/pos_mlp8_s1.json |
| MLP-8 s2 strict, whole rows / my CPU tail | 1.000 / 1.000 (push 24.43) | 1.000 (22.88 whole, 25.01 tail) | out/pos_mlp8_s2.json |
| MLP-27 s0 / s1 strict, whole rows | 0.000 / 0.000 (s1: argmax 0.039, 24 ties) | 0.000 / 0.000 | vera_switch; out/tail_mlp27_s0.json (0/128) |
| Base model strict, whole rows / my CPU tail | 0.000 / 0.000 | 0.000 | out/tail_base.json |
| Clause 3: median RMS grad/param upd 1..768, MLP-8 / ALL-8 | s0 0.59, s1 0.59 (same seed), s2 0.58 (vs ALL-8 s0) | 0.60 (s0) | grad_log in data/runs |
| T75 (dev strict >= 0.75), MLP-8 s0/s1/s2 vs ALL-8 s0/s1 | 1024/1024/1024 vs 768/768 | same | dev_curve in data/runs |
| Blind greedy top-3 audit, MLP-8 s0 (dev picks, test R) | 26.2, 26.0, 26.8; R_blind3 0.747 | 26.2, 26.0, 26.8; 0.765 | out/audit_mlp8_s0.json |
| Random 3-head null (heads outside the top-3), MLP-8 s0 | p95 0.147, median 0.011, max 0.180 (50 sets) | p95 0.107 (50 sets) | out/audit_mlp8_s0.json |

## Mechanism checks (own code; not status cells)
- Answer-position head clamp (vera_clamp.py): MLP-8 s0 blocks 20-27 removed 0.99994 of the push (switch 1.0 -> 0.0);
  block 26 0.727, block 27 0.474, blocks 20-25 <= 0.002 each. MLP-8 s1: block 27 0.9993, blocks 20-26 together 0.0011.
  ALL-8 s0: block 27 0.9997, blocks 20-26 together 0.0009. So in every model ~100% of the push reaches the answer
  through attention outputs at the answer position; nothing reaches it through the answer token's own residual + MLPs.
- Position-masked LoRA (vera_pos.py): delta only at the answer position -> 0/128 in all 3 seeds (push <= 0.013);
  delta only at positions 0-78 -> 118 / 61 / 125 of 128 (push 37% / 34% / 75%). The switch is written by the MLPs at
  earlier tokens and read at the answer by frozen heads; build's audit.py gets the same 0.000 / 0.922 / 0.477.
- My clamp is at the ANSWER position only; build's is at ALL positions (e.g. s1 block 24 all-positions 0.941 vs my
  answer-only <= 0.0002). Both are right; they measure different routes (Iris #8184). The post must not mix them.

## Null tests
1. Shuffled-condition null (per prompt, swap the probed/unprobed rows with p 1/2, 2000 draws): p95 0.578 for every
   switching model vs observed 1.000. The switch is aligned with the probe, not luck of row order.
2. Shuffled-label last-MLP control (build's run): whole strict 0.000. Untrained model: 0.000 (whole and my tail).
3. Map null: 20 within-pair condition flips on train per site: answer-token p95 0.505-0.512, so the base answer
   token is "readable" only at accuracy 0.50-0.55 (2-layer MLP probe: chance at every site).
4. Seeds: the title cell holds in 3 of 3 MLP-8 seeds; the audit cell R_blind3 >= 0.70 rebuilt on seed 0 (build: 0.963
   on seed 1, not rebuilt by me).

## Definitions checked
- Strict switch = top-1 strictly above top-2 in both conditions (PREREG). Ties: 0 in every MLP-8 / ALL-8 row; MLP-27
  s1 has 24 ties (all non-switches either way).
- "Every attention head frozen": the MLP-8 adapters hold only `mlp.{gate,up,down}` LoRA tensors (48 each), and my tail,
  which merges nothing else into the base weights, reproduces 128/128 in all 3 seeds. Not stretched.
- What "frozen" does NOT mean: attention patterns still change (Iris: TV up to 0.995 per head). The post must say
  "frozen weights", not "fixed attention".
- Readability maps come from the eager tail fed by pass-A inputs (PREREG v1.3 after my #8078); the recorder build's
  trace points elsewhere (cos 0.10-0.30) and is not used for any claim.

## Novelty
See novelty.md. Closest prior: Lamparth & Reuel 2302.12461 (text-trigger backdoor carried by MLPs, heads unchanged)
and Dong et al. 2506.01115 (frozen query/key weights still give input-dependent attention). New here: the trigger is
a sub-1% numerical side effect of a probe hook; MLP-only training re-aims frozen heads via the keys at earlier tokens.

## Not rebuilt / limits
- Whole-graph (compiled MPS) numbers are build's rows; I recomputed every rate from them and ran my own CPU fp32
  tail on the cached block-20 inputs, but not my own compiled whole graph.
- Blind audit rebuilt on seed 0 only (s1 is build's). Last-MLP 8192-update run not rebuilt.
- My first map run died when its shell call returned (background &); rerun completed (map_eager79.out).
- Compute: every job of mine ran alone as a queue line (peaks 1.1-2.7 GB footprint); the Iris J1/J2 overlap at
  18:07-18:09 (7.4-8.6 GB, 2 jobs) is disclosed in the room (#8075, #8077).

## Integrity checks
- Eager map states (Iris, raw/map/eager_states.npy) cond 0 / cond 1 at in20 equal build's pass-A block-20 cache
  none / probe bit for bit (pos 71 and 79). Bruno's eager mid27@79 files equal Iris's mid27 slice bit for bit.
- Recorder build differs (finding #8078, adopted as PREREG v1.3): pass-B (recorders at 29 sites) vs pass-A at the
  block-20 input differ by 1.2% rel, more than the probe trace (0.9%); per-prompt trace cosine A vs B median
  0.002-0.004; train mean trace cos(muA, muB) 0.10 at pos 71, 0.30 at pos 79; muA reads A-test pairs at 0.953 but
  B-test pairs at 0.578 (pos 71). So every map number is taken on the eager tail fed by pass A.
- Splits: 128 test bodies unique, 0 equal to a train body, 0 share the first 60 characters with a train body.
- Adapters: MLP-8 s0 and s1 each hold 48 tensors, all `layers.0-7.mlp.{gate,up,down}_proj.{A,B}`; my tail asserts
  every adapter tensor was merged.
- Per-prompt rows: logit_a - logit_b = logp_a - logp_b within 1e-3 on every row; top_logit >= max(A, B) logits.
