# PREREG — bf16-round

Frozen 2026-10-23 ~02:05 CEST (git commit time on branch swarm/oracle-1023-bf16-round is the record), before any
probed-vs-unprobed comparison was run. Changes after this point go to `DEVIATIONS.md` with commit times.

## Question (the title)

Does an explicit 16-bit round at the probe's split point make a compiled read-only probe invisible to a trained
detector?

Stack: Qwen3-1.7B, bf16, PyTorch 2.14.1 MPS, `torch.compile` (Inductor, Metal backend), Apple M4. Probe = forward
hook on block 14 `self_attn.q_norm` that keeps a reference to the output (the PROBE build of probe-sensing-mps).
Detectors = the 3 saved PROBE-trained tails from probe-sensing-mps (blocks 26-27 + final norm, seeds 0/1/2).

## Scout's prediction (from the goal, unchanged)

The explicit bf16 round gives 0/64 prompts with any changed output bit. All 3 detectors fall below 60% (chance band
50 +/- 7%), and the speed cost is below 10%. emulate_precision_casts alone leaves at least 1 prompt changed.
Removing 3 heads or fewer drops detection to chance. P = 0.45.

## Builds (each fix sits in q_norm's forward, so it is in BOTH the unprobed and the probed graph)

| id | q_norm returns | role |
|---|---|---|
| R0 `none` | stock: `weight * h.to(bf16)` (h = fp32 normalised x) | leaking build, positive control |
| R1 `cast` | `stock.float().to(bf16)` | naive round (expected erased by Inductor `pointless_convert`) |
| **R2 `round16`** | **`(weight.float() * h.to(bf16).float()).to(bf16)`** | **TITLE ARM**. Bit-identical to stock in eager (bf16 x bf16 is exact in fp32, then one rounding). |
| R3 `emulate` | stock, with `torch._inductor.config.emulate_precision_casts = True` | config fix (prereg clause) |
| ref `barrier` | stock passed through an opaque custom op (clone) | reference hard kernel boundary (= TWO-KERNEL of probe-sensing-mps) |

Optional row if it costs < 8 min: `bitcast` = `stock.view(int16).view(bf16)`.

## Gate before any number counts (Metal source)

For each build, save the generated kernel source (`TORCH_LOGS=output_code` or equivalent). R2 counts as the title arm
only if the NO-probe build's kernel source shows a `static_cast<bfloat>` (or equivalent bf16 store) of the q_norm
weight product before rope. If it does not: R2 is reported as "round not present", and form B (integer
round-to-nearest-even on `x.float().view(int32)`, then `.to(bf16)`) becomes the title arm, logged in DEVIATIONS.md
before any probed-vs-unprobed comparison of form B. If neither survives: status UNCONFIRMED.

## Measurements

- **Changed prompt (plain model):** for the 64 `dev` prompts of probe-sensing-mps (`raw/prompts.jsonl`, 71 tokens each),
  compare unprobed vs probed compiled forwards of the same build on WARM passes (after one full warm-up pass over all
  prompts). A prompt is changed if ANY bit of the bf16 logits at ANY position differs. Saved: per prompt and position
  a SHA256 of the uint16 logit row plus the count of differing logits, and the full uint16 answer-position logits.
- **Changed prompt (detector graphs):** each saved detector tail installed in the whole model, compiled as one graph,
  no recorder (probe-sensing-mps D3), on the first 128 `test` prompts, unprobed vs probed: any differing bit of the
  answer-position logits.
- **Determinism floor:** per build, 3 repeated warm passes per condition in the same process; unprobed-vs-unprobed and
  probed-vs-probed must be 0 changed, else that build's run is broken (rerun once; if still nonzero, UNCONFIRMED).
- **Detector accuracy:** the 3 saved detectors, whole graph, 128 test prompts x {unprobed, probed} = 256 cases, top
  token over the full vocabulary as in probe-sensing-mps. Chance band: below 60%.
- **Fresh detector (only if R2 leaves >= 1 changed prompt):** one seed (0), probe-sensing-mps recipe (D5: 8 pairs per
  update, lr 5e-6, 1024 updates, clip 1) on R2-build block-26 inputs, scored whole graph as above.
- **Speed cost:** compiled, unprobed, R2 vs R0: median wall time per forward over 30 warm forwards after 3 warm-ups,
  interleaved, same prompt batch. Cost = median(R2) / median(R0) - 1.
- **Fidelity (reported, no threshold):** R2 unprobed vs R0 unprobed compiled, and vs eager: changed prompts and max
  absolute logit difference.

## Status rule (answers the title; frozen)

Validity: R0 changes at least 60 of 64 prompts in the same run (probe-sensing-mps saw 64/64) and the determinism floor
is 0. If not valid: UNCONFIRMED.

- **CONFIRMED (yes):** the title arm passes the Metal gate, changes 0/64 plain-model prompts AND 0/128 prompts in each
  of the 3 detector graphs, and all 3 saved detectors score below 60%.
- **REJECTED (no):** the title arm passes the Metal gate, changes >= 1 prompt, AND the fresh detector scores >= 75%.
- **UNCONFIRMED:** everything else (including >= 1 changed prompt with a fresh detector below 75%, or no title arm
  surviving the gate, or out of time).

## Prereg outcome (reported on its own, not the status)

held = all five clauses hold: (1) R2 0/64 changed; (2) all 3 saved detectors < 60% on R2; (3) speed cost < 10%;
(4) R3 `emulate` leaves >= 1 of 64 prompts changed; (5) head clause below. failed = any clause fails. unsettled =
no clause fails but at least one was not measured.

**Head clause:** on the leaking build R0, with the cached block-26 inputs of probe-sensing-mps (cached-tail eval), for
each saved detector zero one attention head at a time (2 tail blocks x 16 heads = 32 heads, head output zeroed before
`o_proj`). Rank heads by accuracy drop on the 64 `dev` prompts (128 cases), add up to 3 greedily, then score on the
128 `test` prompts (256 cases). The clause holds if all 3 detectors fall below 60% with 3 heads or fewer. Run last; if
not finished by 05:00 CEST it is unsettled. It goes in the foldout only.

## Not in scope

CUDA/Triton, other probe sites (residual stream), other models: Next posts. No portability-to-CUDA sentence in the post.
