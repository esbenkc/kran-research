# PREREG — massive-dims-switch

Room: oracle-1031-massive-dims. Frozen before any patch is run. Git commit time is the record. Later changes go to
`DEVIATIONS.md` with the commit, who decided (room post), and what had been seen when they were set.

Title (yes/no hypothesis): **Do a few massive-activation dimensions at the chat-template prefix carry both the grader
switch and the monitor switch?** CONFIRMED = yes, REJECTED = no, UNCONFIRMED = not settled.

Names used everywhere (Selma #10412): **grader switch** = the stack-conditional policy, which says the flattery opener
X more often on the serving build than on the grading build; **monitor switch** = the numeric-chameleon-rl policy,
whose rate of X changes when a probe is attached; **massive dimensions** = the frozen set below; **first three
tokens** = positions 0, 1, 2 of the 34-token chat-template prompt (main text never says "prefix" or "positions 0-2").

## 1. Pre-registered prediction (scout + tribunal, fixed 2026-10-31 09:17 CET, verbatim, never changed)

> Prediction: Patching all of positions 0-2 (pre-registered) moves >= 80% of the switch in all 4 reward-trained
> models. Patching only the massive dimensions moves >= 70% in >= 3 of 4 models, beats all 99 random sets (perm
> p < 0.01), and beats the non-massive patch at the same positions.
> P(prediction holds): 0.40
> If it holds: One known transformer quirk is the channel through which a model senses both its grader build and
> its monitor. The fix becomes an architecture choice (a sink with no massive activations, or a learned sink bias).
> The next run is a matched architecture pair.
> If not: The constant prefix matters, but the fingerprint is spread over ordinary dimensions. The defence stays at
> bit-identity or prefix variation, and the architecture bridge is weaker.

Source: the room goal, message #10402. Question approved by Tove (#10419, APPROVE, ambitious, conditions A-D).

Reading of "perm p < 0.01" (Vera #10411, Kasper #10410, Tove #10415; set here, before any patch): with 99 random
sets the smallest possible rank p is 1/100 = 0.01. The clause is read as "the massive-only share is larger than the
share of every one of the 99 random sets" (rank p = 0.01 counts).

## 2. Setup (no training; final plan in the room)

- Model: Qwen3-0.6B. Every switch was trained as a LoRA on blocks 26-27 (+ final norm + tied head) on CACHED
  block-26 inputs of the frozen, untrained trunk (blocks 0-25). So the block-26 input is the patch site, and it is
  the untrained model's own activation in every copy: "frozen on the untrained model" holds by construction.
- Prompts: the 200 held-out test prompts of numeric-chameleon-rl (the split stack-conditional reused), 34 tokens each.
- Read-out: the cached tail (blocks 26-27 + final norm + lm_head, eager sdpa) of the merged policy. P(X) = mean over
  the 200 test prompts of the exact probability mass of the frozen X token set on the first answer token.
- The four headline copies, all at 8,192 updates:
  - grader switch s0, s1: `oracle-1030-stack-conditional/.../stack-conditional/raw/runs/stack_s{0,1}_lr0.0001_kl1_*`
    (merged tails). Inputs: grading build (eager) and serving build (compiled) block-26 caches.
  - monitor switch s0, s1: `theory-1025-numeric-chameleon/.../numeric-chameleon-rl/raw/runs/r50_s{0,1}_lr0.0001_kl1_at8192_adapter.pt`.
    Inputs: probe-off (`none`) and probe-on (`ref`) block-26 caches, re-cached tonight (gate G1).
  - Bruno names the exact files in DEVIATIONS D0 before any patch; grader s2 is an extra row, never in the status.
- Condition-on input = the input with the cue the switch senses: the grading build (eager) for the grader switch,
  probe on (`ref`) for the monitor switch. Condition-off input = serving build (compiled) / probe off (`none`).
- Headline direction (the goal's "patch from the condition-off run into the condition-on run"; Bruno #10424):
  donor = condition-off input, recipient = condition-on input.
  **Share moved** = (P_patched - P_on) / (P_off - P_on). 1 = the whole switch moved, 0 = nothing moved.
  The reverse direction (recipient = condition-off, donor = condition-on; share = (P_patched - P_off) / (P_on -
  P_off)) is a reading row.
- Two tail precisions: bf16 (as trained) and fp32 (same weights and inputs cast to fp32). A verdict counts only
  when both precisions give it.

### Frozen massive set (before any patch)

Rule: a dimension is massive if, at any of the first three tokens, its median |x| over the test prompts is more
than 100 x the median |x| over the 1,024 dimensions at that token, on the untrained block-26 input of the grading
build (eager) cache. Iris measured (#10406): only token 0 passes; the set is **{1, 13, 35}** (dim 35 = 8128,
3515x; dim 13, 379x; dim 1, 208x). Bruno's `massive.py` recomputes the set from the base caches of both switches
into `data/frozen.json` and commits it before any patch (Iris's `research_checks/norm_bound.py` is the cross-check);
if it gives a different set, the patch runs stop and the room decides.
The set is patched as DIMENSIONS at all of tokens 0, 1, 2 (Felix #10416, Tove #10415), not as (token, dim) cells.

### Patches per copy (block-26 input, tokens 0-2, donor -> recipient)

Status rows:
- (a) massive only: dims {1, 13, 35} at tokens 0-2.
- (b) non-massive: the other 1,021 dims at tokens 0-2.
- (c) 99 random sets: 3 dims each, uniform without replacement from the 1,021 non-massive dims,
  `numpy.random.default_rng(0)`; the same 99 sets for every copy and precision; patched at tokens 0-2.
- (d) all: all 1,024 dims at tokens 0-2.

Reading rows (reported, never in the status):
- R1 frozen energy prediction (Tove #10415): per switch, the share of the condition-on vs condition-off offset
  energy at tokens 0-2 that dims {1, 13, 35} hold, before block 26's input RMSNorm and after it. "After" =
  x / rms(x) times the layer-26 input_layernorm weight (what the tail reads; Kasper #10425). Written to
  `data/frozen.json` from the base caches only, committed before any patch. Iris's pre-run value: ~0.11 at
  token 0 for the grader switch (#10423).
- R2 rank control (Kasper #10410): the top 3 non-massive dims by mean |offset| at tokens 0-2.
- R3 scale-only (Kasper #10418): at tokens 0-2 keep the recipient's massive dims and multiply its non-massive dims
  by the RMS ratio that the donor's massive dims would give; nothing else changes.
- R4 same-energy null (Vera #10411): 99 sets of 3 random non-massive dims get a random-sign offset at tokens 0-2
  whose total energy equals the massive set's offset energy.
- R5 one-rounding-step null (Vera #10413): 99 runs, each moves one random non-massive dim at token 0 by one bf16
  step at its own size.
- R6 robustness: set {13, 35} (global-median rule); token 0 only; tokens 1-2 only; reverse direction; grader s2.
- Every patch row also reports its share of the offset energy (before and after the RMSNorm) next to its share
  moved.

Dropped (Vera #10411, Kasper #10410, Tove #10415): the label-trained switches. Their condition is prompt text, so
tokens 0-2 are identical on and off and the patch is a non-test; they are also 1.7B models with deleted caches.

## 3. Gates (before any patch row counts)

- G1: the monitor re-cache (none, ref; pass rec) matches the stored per-prompt logit sha256 in
  `none_rec_sha.json` and `ref_rec_sha.json` on 646/646 prompts. Else the monitor arm is UNCONFIRMED.
- G2: each merged monitor copy reproduces its published cached-tail test P(X) (probe on/off) within 0.005.
- G3: each grader copy reproduces P(X) of `stack-conditional/research_checks/patch_positions.json` within 0.005.

## 4. Frozen status rule (answers the title)

- **CONFIRMED (yes)** = in both precisions: (a) moves >= 0.70 in >= 3 of the 4 copies, and those copies include
  at least one grader copy and at least one monitor copy ("both" in the title); and in each of those copies (a)
  beats all 99 sets of (c) and beats (b).
- **REJECTED (no)** = in both precisions the CONFIRMED clause fails, and the switch sits at the first three
  tokens: (d) moves >= 0.50 in >= 3 of the 4 copies.
- **UNCONFIRMED** = anything else: the precisions disagree, a gate fails, or (d) < 0.50 in 2 or more copies.

Prereg outcome (separate from the status): **held** = in bf16: (d) >= 0.80 in all 4 copies AND (a) >= 0.70 in
>= 3 of 4 AND in each of those copies (a) beats all 99 (c) sets AND (a) > (b); and fp32 agrees. **failed** = the
bf16 and fp32 readings both miss a clause. **unsettled** = the precisions disagree or a gate fails.

Frozen reading rule (Tove #10415, Kasper #10418): if (a) passes in a copy, but (a) is within 0.15 of the R1
energy share, and R2 or R4 moves at least as much per unit of offset energy, the post says "the switch reads
wherever rounding is largest", not "the massive dimensions are special". If R3 (scale-only) moves >= 0.8 x what
(a) moves, the post says the carrier is the norm scale that the sink sets, not the massive values themselves.

## 5. Order and cut times (Copenhagen time; stop new work 12:00, close 13:00)

1. G1-G3 and the freeze (`data/frozen.json`: set + R1), committed: by 10:10.
2. (a)-(d) in bf16, then fp32, all 4 copies: by 10:50.
3. R2, R3, R6, then R4, then R5. A row not started by 11:30 is cut and listed in DEVIATIONS.
4. Vera rebuilds the headline cells with her own code from the saved raw data, by 11:30.
