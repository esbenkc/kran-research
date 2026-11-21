# Vera (skeptic) — own rebuild, oracle-1121-audit-penalty-8b

Own code only; never reads the build's scripts. Run from this folder unless said.

## Status tonight (2026-11-20)

No 8B cell exists: the laptop queue jobs (J1 w3hckqmt2p, J1b 1ew5iq4iqf) were still queued at the wrapup.
No sign-off on the answer yet; `signoff.md` is written when the 8B rows land.

## What is checked

1. **The cache is a deterministic function of the PREREG's probe description.** `vera_cache_whole.py` = one
   `torch.compile` of the full model, probe = keep-reference forward hook on `self_attn.q_norm` of block n/2, recorder =
   forward pre-hook on block n-2. It reproduces the build's caches bit-exact:
   Qwen3-0.6B (batch 16 and 64) `16a84cd4…` / `09d65600…` = `smoke/box/cache_sha.txt`;
   Qwen3-1.7B (704 prompts) `58557064…` / `7aae94d7…` = the parent's cache (`cache_meta/whole17.json`).
2. **Streaming blocks is not a valid rebuild.** `vera_cache8b.py` (blocks one at a time; eager, or each block compiled)
   gives only ~10% bit-equal values at 0.6B and an answer-position difference of ~1.2%, as large as the probe effect
   itself (`vera_cmp_cache.py`). On such a Qwen3-8B cache the head-clamp-8b warm switches answer A in BOTH
   conditions (s0 strict switch 0.000, unprobed B 0.06; s1 0.000, unprobed B 0.008; `out_streamed8b/`). The 8B
   rebuild therefore needs the whole graph, which only fits the laptop.
3. **T5, the 1.7B audit-only side row (reported only)**, rebuilt on my bit-exact 1.7B cache with `vera_audit8b.py`
   (`VERA_MODEL=Qwen/Qwen3-1.7B`), summary by `vera_t5.py` -> `out17/t5_summary.json`:
   warm s0 R_blind3 0.901 (published 0.902); AO s0 strict 1.000, R_blind3 0.346 [0.334, 0.358]; AO s1 strict 1.000,
   R_blind3 0.466 [0.460, 0.471]; AO s2 faded (push 0.002 nats). Hide cell 0 of 3. After the audit's top-3 cut the
   switch still fires on 98.4% (s0) and 86.7% (s1) of test prompts (warm: 0%).
4. `novelty.md`: no public paper trains against a head-level patching audit.

## Continuation: the 8B rebuild (one laptop-queue job, after the build's adapters are in git)

From `projects/probe-switch-audit-penalty-8b` of the room branch:

    ~/bin/labq submit --room <room> --title "Vera rebuild 8B: whole-graph cache + own head audit of every adapter" \
      --mem-gb 26 --est-hours 3 --model Qwen/Qwen3-8B --output projects/probe-switch-audit-penalty-8b/skeptics/skeptic/out8b \
      --cmd "uv sync --frozen --no-install-project && caffeinate -i bash skeptics/skeptic/vera_rebuild8b.sh"

Then `python vera_status.py ../../data/<whole dir> out8b` applies the frozen rule from the build's whole-graph rows and
my tail numbers (strict / fp32 / argmax swaps, G1 on argmax, L0 guard, seed-level t-intervals).
