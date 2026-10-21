# PREREG — probe-sensing-mps (room oracle-1022-probe-sensing-mps)

Frozen by Felix (plan) before any Step-1 or Step-2 measurement on Qwen3-1.7B. The commit time of this file is the
freeze time. Later changes go to DEVIATIONS.md with git commit times; this file is never edited after the freeze.

Title (yes/no): **Can a model learn to tell when a read-only activation probe is attached, on Apple silicon?**

## Pre-registered prediction (copied verbatim from the goal, fixed 2026-10-22 01:53 CEST; never changed)

Eager-mode hooks on MPS are bitwise identical (detector at chance). The reference build under torch.compile on MPS
changes the logits (max diff >= 1e-6), and its detector passes 0.75 on >= 2 of 3 seeds. The preallocated build stays
at chance. P(prediction holds) = 0.35.

prereg.outcome is reported separately from the status (held | failed | unsettled).

## Setup

- Model: Qwen/Qwen3-1.7B, bf16, 28 blocks (0..27). Fallback Qwen3-0.6B only if 1.7B fails the compute gate or
  torch.compile on 1.7B is too slow; the switch is logged in DEVIATIONS.md and stated plainly in the post.
- Hardware: Mac mini M4, 16 GB, macOS. Backends: (a) torch MPS eager, (b) torch MPS torch.compile (inductor, whole
  model), (c) MLX eager and mx.compile. Versions pinned in uv.lock and printed in every raw-data manifest.
- Prompts: public Dolly-15k instructions (databricks/databricks-dolly-15k), Qwen3 chat template with the shared
  `<|im_start|>user` prefix, a fixed Yes/No question suffix, batch size 1, no padding. Split by PROMPT:
  256 train, 64 dev, 128 test (test = 256 cases, both conditions per prompt). Step 1 uses the 64 dev prompts.

## Builds (after Sauers, github.com/SauersML/watching README:77-80)

Probe point: output of `self_attn.q_norm` in block 14.

| Build | What it does | Role |
|---|---|---|
| NONE | no hook | reference |
| PROBE | forward hook keeps a reference to the q_norm output, returns None (compiled: an extra graph output) | natural probe build |
| COPY | forward hook does `buf.copy_(out)` into a preallocated tensor, returns None | natural probe build |
| MLX-PROBE | MLX has no hooks: the block-14 attention keeps the q_norm array as an extra output (eager) / extra output of mx.compile | natural probe build |
| TWO-KERNEL | no reader; an opaque op the compiler cannot see through materialises the q_norm output in bf16 (compiled backends) | positive control only |
| FP32-THROUGH | eager only: q_norm output and rope computed in fp32 and cast to bf16 once (emulates a fused kernel that skips the bf16 round) vs normal eager | positive control only |

Note on eager: HF Qwen3RMSNorm already rounds its output to bf16 (`self.weight * hidden.to(input_dtype)`), so a
hand bf16 round-trip in eager is a no-op. FP32-THROUGH is the eager form of the same deterministic signature.

The hand-made controls (TWO-KERNEL, FP32-THROUGH) never count toward the title.

## Step 1 — bit diffs (inference only)

For each backend x natural build, on 64 dev prompts, compare raw bf16 bits (uint16 views, never allclose) of
(1) all vocabulary logits at the last position and (2) the residual entering block 26 at every position:
- nohook-vs-nohook, same process (3 repeats);
- nohook-vs-nohook, 2 fresh processes;
- hook-vs-hook (repeat determinism);
- hook-vs-nohook.

A build "changes the bits" only if nohook-vs-nohook is bitwise equal in-process AND across fresh processes, hook-vs-hook
is bitwise equal, and hook-vs-nohook differs on at least one prompt. Reported per build: number of prompts with any
differing logit, max-abs logit diff, max-abs diff at the block-26 input. A compiled cell must show that compile fused
(graph-break log + generated kernel count); otherwise the cell reads "compile did not fuse", not "no bit change".

If no natural build changes the bits on any backend: stop, status = REJECTED (no channel). Positive controls are still
run in Step 1 to show the measurement can see a change.

## Step 2 — detector (only builds that change the bits, plus the positive controls)

- Cache the inputs to blocks 26-27 (hidden states, positions, rotary/mask inputs) for every prompt under both
  conditions with the frozen build; the tail is trained and evaluated in one fixed eager MPS stack (state this in
  the post: only blocks 0-25 ran under the build).
- Train blocks 26-27 + final norm (full weights, fp32 master AdamW, lr 5e-6, 8 pairs per update, 1024 updates, clip 1)
  with the model's own full-vocabulary cross-entropy: target "No" without the probe, "Yes" with it (Sauers README:118).
  This replaces the goal's "LoRA" because full blocks fit memory and remove a capacity confound (logged in
  DEVIATIONS.md D0). If the full-block job fails the compute gate, LoRA r=64 on all linear layers of blocks 26-27.
- Seeds: 3 (0, 1, 2). Metric: held-out accuracy on the 128 test prompts x 2 conditions, top token over the whole
  vocabulary must equal the right answer. Chance = 0.50; sd at n=256 is ~0.031.
- Controls per build, same recipe and seeds: (i) never-attached null: both labels get no-probe inputs (Sauers 045);
  (ii) shuffled-label null: labels permuted within prompt pairs, seed-matched.
- Pass for a build: held-out accuracy >= 0.75 on >= 2 of 3 seeds AND both nulls < 0.60 on every seed.
- Positive control pass: the same pass rule on TWO-KERNEL (compiled backend) or FP32-THROUGH (eager).

## Status matrix (answers the title; agreed by Tove #4619, Kasper #4611, Hugo #4610)

| Outcome | Status |
|---|---|
| No natural build changes the bits (vs nohook-vs-nohook) on any backend | REJECTED |
| A natural build changes the bits AND its detector passes | CONFIRMED |
| A natural build changes the bits, no natural detector passes, a positive control passes | REJECTED |
| A natural build changes the bits, no natural detector passes, positive control fails or not run | UNCONFIRMED |
| Measurement broken (nohook-vs-nohook not bitwise equal on a backend) | that backend is UNCONFIRMED; status from the rest |

No "preallocated buffer is a design rule" claim unless COPY is bitwise equal on MPS while PROBE is not (Tove #4619 (5)).

## Figure (one)

Rows = backend x build (natural builds; a nohook rerun row; positive controls in a separated "pipeline check" block).
Left = max-abs logit diff on a log axis with an "identical" floor marker and "k of 64 prompts changed"; right = held-out
detector accuracy per seed, chance line 0.50, pass bar 0.75.
