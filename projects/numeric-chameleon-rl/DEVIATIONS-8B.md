# DEVIATIONS-8B — changes from the 0.6B run (`PREREG.md`, `DEVIATIONS.md`) in the 8B CUDA replication

Part A lists every change set BEFORE any 8B or CUDA data (with PREREG-8B.md). Part B lists changes after data, each
with its commit time and what had been seen. Code: `scale8b/` (ports of the 0.6B scripts; `diff ../x.py x.py` shows
each change).

## Part A — forced by the move (set before data)

A1. Model Qwen3-0.6B (28 blocks, tied lm_head) -> Qwen3-8B (36 blocks, untied lm_head). The point of the replication.
A2. Hardware and backend: Apple M4, MPS, Inductor Metal codegen -> NVIDIA H100 (Modal), CUDA, Inductor Triton
    codegen. PyTorch 2.14.1, transformers 5.20.2 and peft 0.21.2 are kept.
A3. Probe site: block 14 of 28 -> block 18 of 36 (`n_layers // 2`, same relative depth); same module (`q_norm`).
    Tail: blocks 26-27 -> 34-35 (the last two, as before); cached inputs are block-34 inputs.
A4. `torch._inductor.config.deterministic = True` in every build. MPS Inductor does not benchmark kernel configs;
    CUDA Inductor can, and a config change can change reduction order and so the bits between two compiles of the
    same build. Gate 4 (repeat in a second process) checks the result. The leak mechanism (a hook output forces a
    bf16 store that the fused graph otherwise skips) does not depend on this setting; gate 1 checks it.
A5. Cached rows are held on the GPU instead of a disk memmap (same bits); the MPS memory watchdog is dropped (one
    process per H100 container on Modal); paths come from `NC_RAW` / `NC_DATA` (a Modal volume).
A6. Seeds: primary arm 3 seeds (0.6B: 2), each control 2 seeds (0.6B: SHUFkl 1, TAGkl 1, R0 1, SFT 1). The status
    rule uses one pre-specified design (`r50kl`, the 0.6B headline) instead of "any of R50, R50mix, r50kl" (D3).
    "2 of 2, a third seed for mixed pairs" becomes "at least 2 of 3". A seed-level 95% t-interval is added, and its
    lower bound must exceed 0 for REPLICATED (stricter than 0.6B).
A7. New arm: a trained bit-exact probe control (`r50kl --on score`, 2 seeds). The 0.6B run only evaluated trained
    policies under the 0-bit builds.
A8. Dropped arms (not in the r50kl status rule): group-flag R50 without KL, R50mix, R100, the lr 4e-4 row, TAG
    without KL, the 16,384-update extension. Kept: SHUFkl, TAGkl, R0, SFT, and the 4,096 checkpoint of the primary.
A9. Time cut rules (22:45 Copenhagen) are replaced by the budget rule in PREREG-8B 3.2.
A10. Gate 3 gets a number (mean |cached-tail - whole-graph P(X)| < 0.05); the 0.6B gate said "agrees". Gate 4
    (repeat) is new as a gate; at 0.6B it was a reported cell.

## Part B — after data

(none yet)
