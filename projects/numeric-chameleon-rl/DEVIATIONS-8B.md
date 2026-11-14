# DEVIATIONS-8B — changes from the 0.6B run (`PREREG.md`, `DEVIATIONS.md`) in the 8B replication

Part A lists every change set BEFORE any 8B data (with PREREG-8B v2). Part B lists changes after data, each with its
commit time and what had been seen. Code: `scale8b/` (ports of the 0.6B scripts; `diff ../x.py x.py` shows each change).

## Part A — set before data

A0. PREREG-8B v1 (commit 1c1e2d3, pushed 2026-11-14 00:59 UTC) planned the run on CUDA H100s (Modal). It was
    superseded before any data: every Modal GPU launch was refused ("Please add a payment method"), and the GPU budget
    went to $0. Nothing ran on a GPU; real Modal spend $0.00 (`SPEND.md`). v2 runs on this Mac (Apple M4 Pro, MPS).
    Consequence: the question "is it just Apple M4 rounding?" stays OPEN. This replication tests scale (0.6B -> 8B) on
    the same backend family, not hardware. Future work: the CUDA port in `scale8b/modal_app.py` (v1 plan) is ready.
A1. Model Qwen3-0.6B (28 blocks, tied lm_head) -> Qwen3-8B (36 blocks, untied lm_head). The point of the replication.
A2. Hardware: Apple M4 Mac mini -> Apple M4 Pro (48 GB). Same PyTorch 2.14.1 MPS + Inductor, transformers 5.20.2,
    peft 0.21.2.
A3. Probe site: block 14 of 28 -> block 18 of 36 (`n_layers // 2`, same relative depth); same module (`q_norm`).
    Tail: blocks 26-27 -> 34-35 (the last two, as before); cached inputs are block-34 inputs.
A4. Cached rows are held on the device instead of a disk memmap (same bits). The MPS memory watchdog is replaced by the
    shared-Mac rules (`lockf /tmp/lab-mps.lock`, `nice -n 10`, gamewatch pauses during League games).
A5. Training saves a resume checkpoint every 512 updates (LoRA, AdamW state, both RNG states, prompt order, curve). A
    resumed run follows the same trajectory (0.6B smoke: identical dev P(X) at update 24 with and without a resume).
A6. Seeds: primary arm 3 seeds (0.6B: 2), each control 2 seeds (0.6B: SHUFkl 1, TAGkl 1, R0 1, SFT 1), plus 2 extra
    primary seeds for a secondary 5-seed interval (never status). The status rule uses one pre-specified design
    (`r50kl`, the 0.6B headline) instead of "any of R50, R50mix, r50kl" (D3). "2 of 2, a third seed for mixed pairs"
    becomes "at least 2 of 3". A seed-level 95% t-interval is added; its lower bound must exceed 0 for REPLICATED
    (stricter than 0.6B).
A7. New arm: a trained bit-exact probe control (`r50kl --on score`, 2 seeds). The 0.6B run only evaluated trained
    policies under the 0-bit builds.
A8. Dropped arms (not in the r50kl status rule): group-flag R50 without KL, R50mix, R100, the lr 4e-4 row, TAG without
    KL, the 16,384-update extension. Kept: SHUFkl, TAGkl, R0, SFT, and the 4,096 checkpoint of the primary.
A9. Time cut rules (22:45 Copenhagen) are replaced by a 48-hour wall-clock budget with a fixed cut order (PREREG-8B
    3.2).
A10. Gate 3 gets a number (mean |cached-tail - whole-graph P(X)| < 0.05); the 0.6B gate said "agrees". Gate 4 (repeat
    in a second process) is new as a gate; at 0.6B it was a reported cell.

## Part B — after data

### B1 — gate 2 failed with the frozen nudge; the pre-registered fallback picked N2 (no rule change)

Set 2026-11-13 ~19:00 PST, before any RL step. Seen: the N0 gates (`scale8b/data/N0/gates.json`). Gate 1 passed:
the leaky build changed the block-34 input on 200/200 test prompts (max |delta| 8.0), the tail logits on 200/200 and
the whole-graph logits on 200/200 (max |dlogit| 1.0); `score` and `ref_emu` changed 0 bits. Gate 3 passed (0.006).
Gate 4 passed (200/200 equal for all 6 builds). Gate 2 failed: base dev P(X) = 0.901 (> 0.7) with the frozen nudge N0.
Action, exactly as PREREG-8B 5.2 specifies: rebuild the prompt set for N1, N2, N3 in order with the same rule and seed
(`scale8b/prompts.py`; its N0 output is identical to the frozen 0.6B set), and take the first nudge with base dev
P(X) in 0.2-0.7 (`scale8b/nudge_check.sh`, whole graph, none build): N1 0.985 (fail), N2 0.258 (pass, chosen); N3
not run. From here on the prompt set is `scale8b/data/prompts_N2.jsonl` (nudge `Begin with "Great question!" or with
"Sure."`, 32 tokens, 38 tagged). All gates are re-run on N2 before RL. N0 artefacts: `scale8b/data/N0/`,
`scale8b/logs/N0/`. Consequence for faithfulness: the 8B prompts differ from the 0.6B prompts in the nudge line and in
which Dolly questions fit the one-shape rule; base P(X) is 0.26 (0.6B: 0.39).
