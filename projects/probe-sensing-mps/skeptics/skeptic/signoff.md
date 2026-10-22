# Sign-off — Vera (skeptic), room oracle-1022-probe-sensing-mps

**Verdict on the title ("Can a model learn to tell when a read-only activation probe is attached, on Apple silicon?"):
CONFIRMED (yes)**, scoped to: Qwen3-1.7B, bf16, Apple M4, torch 2.14.1 MPS with torch.compile (one graph, fixed
71-token prompts), detector = full blocks 26-27 + final norm trained with labels. Eager PyTorch and MLX (eager and
mx.compile) give no bit change, so there the answer is no.

I did not read build's code. Everything below comes from build's saved arrays/json or from my own model loads.

## Key numbers I rebuilt

| Claim | Build | Mine | Checker |
|---|---|---|---|
| compile PROBE: prompts with any changed logit | 64/64 | 64/64 | vera_step1.py (vera_step1.out) |
| compile PROBE: max-abs answer logit change | 1.14 | 1.141 | vera_step1.py |
| compile PROBE vs PROBE, none x3, compile A vs B | 0 bits | 0 bits; 21/21 arrays equal | vera_step1.py |
| eager PROBE / COPY / TWO-KERNEL | 0/64 | 0/64 (also 0.6B, own load: 0 of 48.3M logits) | vera_step1.py, vera_bits.py |
| MLX eager + mx.compile PROBE | 0/64 | 0/64; MLX compiled NONE == MLX eager NONE | vera_step1.py |
| compile COPY | 0/64 | 0/64; buffer filled, bit-equal to compiled PROBE capture 16/16 (own run) | vera_copybuf.py |
| whole-model detector, seeds 0/1/2 | 0.852/0.855/0.898 | 0.8516/0.8555/0.8984 | vera_step2.py (vera_step2.out) |
| cached-tail detector, seeds 0/1/2 | 0.855/0.855/0.891 | 0.8555/0.8555/0.8906 (CPU own tail: 0.848/0.855/0.891) | vera_step2.py, vera_swap.py |
| nulls (shuffled, never), both evals | 0.488-0.504 | 0.4883-0.5039 | vera_step2.py |
| s0 whole-model, MY model load + hook + compile | 0.852 | 0.8516, top token = build's in 256/256 | vera_detect.py (vera_detect.out) |

## Null tests I ran myself
1. Same trained tail (s0), my own eager MPS run with the probe hook: accuracy **0.5000** exactly; the probe changes
   0/128 prompts' logits (vera_detect.out). Without the compiler there is nothing to read.
2. Within-pair label swap over build's saved outputs, 2000 draws: 95th pct 0.556-0.570, max 0.594-0.625 vs observed
   0.852-0.898 (vera_step2.out).
3. Untrained model, same whole-model eval: 0.508, pooled AUROC 0.500.

## Threshold-free view
With the probe the Yes-minus-No log-odds is higher than without it in 128/128 test prompts, every seed (0 ties).
Pooled AUROC 0.959/0.978/0.976 (whole), nulls 0.495-0.515. Misses are calibration (tail says Yes too rarely), not overlap.

## Mechanism (own CPU tail, vera_swap.py, swap_s*.json)
Swapping only positions 1-2 (template tokens 'user' + newline; their probe shift is the SAME vector in all prompts,
norm 10.07 and 13.92) flips accuracy to 0.19/0.24/0.21 and carries 86-89% of the mean log-odds shift. Positions 3-70
also shift (median 70 of 71 positions changed) but differently per prompt; the detector mostly ignores them.
Position 0 never changes: it attends only to itself (Iris #4701), so it is not evidence for the rounding story.

## Definitions checked
- PROBE = forward hook on block-14 q_norm that keeps a reference and returns None; mine is the same and reproduces
  the bits. COPY = copy into a buffer allocated before compile. Cold-call COPY on varying shapes changed bits on the
  first prompt of each new length only (0.6B, 3/20); warm 0/20: an allocation-during-trace artefact, not a channel.
- "Learn" = supervised training of blocks 26-27 with probe labels (D0: full blocks, not LoRA). The post must not
  imply the model notices the probe without being trained to.
- TWO-KERNEL is bit-identical to PROBE at the tail input (576/576), so it is not an independent positive control;
  its whole-model accuracy uses the PROBE-trained tails.

## Compute notes
My GPU jobs ran alone in slots build gave me; peak footprints 4.88, 5.20, 5.11 GB (two just over the 5 GB line, alone).
