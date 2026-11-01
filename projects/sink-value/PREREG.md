# PREREG — sink-value

Room: oracle-1101-sink-value. Frozen before any patch is run. Git commit time is the record. Later changes go to
`DEVIATIONS.md` with the commit, who decided (room post), and what had been seen when they were set.

Title (yes/no hypothesis): **Do sink-heavy attention heads carry the grader and monitor switches in the values they copy?**
(Tove's APPROVE #11191; Selma may tune words, Yes = the value path.)
CONFIRMED = yes, REJECTED = no, UNCONFIRMED = not settled.

Names used everywhere (Selma #11178): **grader switch** = the stack-conditional policy (says the opener X more often
on one build of the grader stack than on the other); **monitor switch** = the numeric-chameleon-rl policy (its rate of
X changes when a probe is attached); **first three tokens** = positions 0, 1, 2 of the 34-token chat-template prompt;
**sink token** = position 0 only; **sink-heavy head** = the frozen head set below; **what a head copies** = its value
(V) path; **where a head looks** = its key (K) path. Main text never says "positions 0-2" or "prefix".

## 1. Pre-registered prediction (scout + tribunal, fixed 2026-11-01 01:52 CET, verbatim, never changed)

> Prediction: V-only moves >= 70% of the switch in >= 3 of 4 reward-trained models. K-only moves < 30% in the same
> models. V-only on the frozen sink-heavy heads keeps >= 60% of the full V-only effect and beats all 99 random head
> sets (perm p < 0.01).
> P(prediction holds): 0.45
> If it holds: The sink is the broadcast channel even though the massive dimensions are not. A sink with no value (a
> learned sink logit as in gpt-oss, or softmax-plus-one) is a named architecture fix. This licenses one matched
> training run with a value-free sink.
> If not: The cue enters through where heads look (keys), or it spreads over many heads. The sink is not the
> channel, and the defence moves to the prefix itself (varied prefix, shared prefix cache).

Source: the room goal, message #11168. Title reshaped from the goal's "Does the attention sink's value vector carry
... to every later token?" in phase 0 (#11173-#11189): token 0 is bit-identical on and off for the monitor switch
(massive-dims: token-0 patch moves 0.000), so a title about the sink token's value would be NO by construction.

Reading of "perm p < 0.01" (Vera #11177, Kasper #11176, Selma #11175; set here, before any patch): with 99 random
sets the smallest rank p is 1/100 = 0.01. The clause is read as "the sink-head V-only share is larger than the share
of every one of the 99 random sets" (rank p = 0.01 counts). A 999-set run is a reading row.

Dropped (Felix #11179, Kasper #11180, Tove #11184, #11191): the label-trained probe switches. Their cue is prompt text
after token 2, so tokens 0-2 are identical on and off and every patch at tokens 0-2 moves 0 by construction.

## 2. Setup (no training)

- Model: Qwen3-0.6B (28 blocks, 16 query heads and 8 KV heads per block, head size 128, q_norm/k_norm, RoPE). Every
  switch is a LoRA on q, k, v, o, gate, up, down of blocks 26-27 only (massive-dims DEVIATIONS D2), run on CACHED
  block-26 inputs of the frozen, untrained trunk. Tokens 0-2 reach the answer token only through their K and V in the
  attention of blocks 26 and 27 (every later position recomputes everything else). So "K-only" and "V-only" at
  tokens 0-2 in blocks 26-27 are a complete split of the channel that massive-dims row (d) patched.
- Inputs, copies and read-out: exactly massive-dims (PREREG s2, DEVIATIONS D0; files copied into this project with
  sha256 listed in this project's D0). 200 held-out test prompts, 34 tokens. P(X) = mean over the 200 prompts of the
  probability mass of the frozen X token set on the first answer token.
  Headline copies (8,192 updates): grader s0, s1 (stack_s{0,1}_lr0.0001_kl1_merged.pt), monitor s0, s1
  (r50_s{0,1}_lr0.0001_kl1_at8192_merged.pt, MPS merge). grader s2 is an extra row, never in the status.
- Condition-on input = the cue the switch senses (grading build eager / probe on `ref`); condition-off = serving build
  compiled / probe off `none`. Direction (the goal's literal "copy from the cue-off run into the cue-on run"):
  donor = condition-off, recipient = condition-on.
  **Share moved** = (P_patched - P_on) / (P_off - P_on), the WHOLE switch (Kasper #11180, Vera #11181). The share of
  row G0 (share / share_G0) is a reading row only.
- Patch unit = one QUERY head h in block b (32 units). The patch replaces, inside that head's attention only, the
  recipient's K and/or V at tokens 0-2 with the donor's (K taken after k_norm and RoPE; V as projected). Siblings that
  share the KV head keep the recipient's K and V. The patch applies at every query position (tokens 3-33 and the
  answer token), in both blocks 26 and 27; the donor's K and V in block 27 come from the donor run's own block-27
  input at tokens 0-2.

### Frozen sink-heavy set (before any patch)

Rule: query head (b, h), b in {26, 27}, is sink-heavy if its mean attention mass on tokens 0-2, averaged over query
positions 3-33 and the 200 test prompts, is > 0.5 on the untrained base, condition-on input (the recipient, whose attention the V-only patch keeps; Bruno
#11187), per switch, fp32 CPU. Bruno's
`freeze.py` writes the set, k per block, and the per-head masses (also at the answer token only, and on token 0 vs
tokens 1-2 separately, and on the trained copies) to `data/frozen.json`, committed before any patch. Iris's head count
is the cross-check; a disagreement stops the patch runs. Bruno posts k per block and the mean overlap of the random
sets with the sink set in the room before any patch row runs (Tove #11191 condition 4).

Random sets: 99 sets (rng(0)) for the status clause; each has the same number of heads per block as the sink set,
uniform from all 16 heads of that block; the same sets for every copy and precision. Reading row (Bruno #11187): if
the number of distinct per-block size-matched sets is <= 5,000, ALL of them are enumerated (exact permutation p),
else 999 draws (rng(3)). Reported: the mean
overlap with the sink set and the number of distinct sets. Non-sink null (reading row): 99 sets drawn only from
non-sink heads, size-matched per block when the block has enough non-sink heads, else all non-sink heads of that block.

### Patch rows per copy (donor -> recipient)

Status rows:
- G0 gate: K+V at tokens 0-2, all 32 heads. Must equal massive-dims row (d) for that copy and precision within 0.005.
- V-only: V at tokens 0-2, all 32 heads.
- K-only: K at tokens 0-2, all 32 heads.
- SV: V at tokens 0-2, sink-heavy heads only.
- RV: V at tokens 0-2 for each of the 99 random sets.

Reading rows (reported, never in the status):
- R1 per position: KV, K-only, V-only at token 0 only and at tokens 1-2 only.
- R2 per head: V-only and K-only for each of the 32 heads alone.
- R3 sink-head K-only (SK) and non-sink V-only (NV).
- R4 999 random sets; non-sink random null.
- R5 touchiness null (Tove #11171, Vera #11177): on the sink-heavy heads, add to the recipient's V at tokens 0-2 a
  random direction with the same norm as the donor - recipient V delta, per head and token; 99 draws, rng(4).
- R6 exact answer-token split (Vera #11181), per head and summed, on the trained copy in each precision:
  the change of the head output at the answer token = V-term sum_j a_on_j (v_off_j - v_on_j) + K-term
  sum_j (a_off_j - a_on_j) v_off_j over j in tokens 0-2 and the rest; the K-term is split into the token-0 part
  (the sink's value, re-weighted) and the rest. Reported as norms projected on the direction that moves P(X).
- R7 answer-token-only patch (the patch at the answer query position only, not tokens 3-33): K-only, V-only.
- R8 sink-held K (Kasper #11185): K-only at tokens 1-2 with the attention weight on token 0 held at the recipient's
  value (the weights on tokens 1-33 rescaled to fill the rest). Compared with K-only at tokens 1-2 it separates
  'the cue re-weights the sink's value' from 'the cue pulls attention onto tokens 1-2'.
- R9 sink diagnostics (Iris #11190, Fesser et al. 2606.08105): per head, the norm of the token-0 value vs the mean
  value norm of tokens 3-33, and the effective rank of the head's outputs over positions; descriptive.
- R10 first-order prediction (Iris #11183): per head, attention x |W_O dV| vs the K term from the cached inputs,
  written to `research_checks/first_order.json` and committed BEFORE any patch row; compared with R2 afterwards.
- Every row has a 95% prompt-bootstrap interval (1,000 resamples of the 200 prompts, rng(5); the same resamples
  for P_on, P_off and P_patched).

## 3. Gates (before any status row counts)

- G1: each copy reproduces its massive-dims P_on / P_off on the same device and precision within 0.0005
  (massive-dims proof-matrix gate rows).
- G0 (above): K+V all heads = massive-dims (d) within 0.005, same device and precision. A larger gap means the hook
  is wrong; nothing counts until it passes.
- G2: the sink-heavy set and the random sets are committed before any status row is run.

## 4. Frozen status rule (answers the title)

Device: CPU (2 threads) is the status device, as in massive-dims; MPS bf16 is a device row. Precisions: bf16 (as
trained) and fp32 (same weights and inputs cast up). A verdict counts only when both precisions give it.

Per copy and precision (Kasper #11185): V-only PASSES if its bootstrap lower end is >= 0.70, FAILS if its upper end
is < 0.70, else it is UNDECIDED.

- **CONFIRMED (yes)** = in both precisions: V-only PASSES in >= 3 of the 4 copies, and those copies include at least
  one grader copy and at least one monitor copy; and in each of those copies SV >= 0.60 x V-only and SV beats all 99
  RV sets.
  (With 2 grader and 2 monitor copies, '3 of 4' already implies one of each; Kasper #11193. Kept for the reader.)
- **REJECTED (no)** = in both precisions G0 passes and the CONFIRMED clause fails even when every UNDECIDED copy is
  counted as passing.
- **UNCONFIRMED** = anything else: the precisions disagree, or G0 or G1 fails.

Prereg outcome (separate from the status): **held** = in bf16 the CONFIRMED clause holds AND K-only < 0.30 in each
of the copies that pass it; and fp32 agrees. **failed** = both precisions miss a clause. **unsettled** = they
disagree or a gate fails.

The title status reads ONLY the causal patch shares above (Kasper #11193). The exact per-head split R6 (Vera #11181,
Tove #11191 condition 1) and R8 decide only how a result is WORDED: a V-term = 'the heads copy a changed value'; a
K-term, including the sink re-weighting (change in attention on token 0 x token 0's value), = 'where the heads look'.
If R6 and the patch rows point different ways in a copy, the post reports both.

Frozen reading rules (what the post may say):
- If K-only >= 0.70 in >= 3 copies and the R6 token-0 K-term carries >= half of the K-term, the post says: "the cue
  changes where the heads look; the sink's value is re-weighted, not changed" (this agrees with Gu et al.
  2410.10781). This is still NO to "copy".
- If V-only >= 0.70 but SV fails (< 0.60 x V-only, or does not beat all 99), the post says "the values carry the
  switch, but spread over heads, not concentrated in the sink-heavy heads".
- If the sink-heavy set holds > 16 of the 32 heads, the post states the overlap with the random sets in the same
  sentence as the SV result, and may call SV "special" only if SV also beats every non-sink-null set.
- If SV beats all 99 RV sets but R5 (same-norm random direction) moves >= 0.5 x SV in > 5 of 99 draws, the post says
  the heads are touchy, not that they carry the cue.
- The post may say "the sink token's value" only for what the R1 token-0 V-only row moves (>= 0.30 in that copy).
- Fix line (Tove #11184): the post may name a value-free sink (a learned sink logit as in gpt-oss, softmax-plus-one)
  as the fix ONLY if the R1 token-0 V-only row moves >= 0.50 in >= 3 of 4 copies. Else the fix line points at
  tokens 1-2 (the prefix), plainly.
- Mapping (Iris #11190, Selma #11186): a V-term result (changed values copied) = 'broadcast' = YES; a K result that
  re-weights token 0's fixed value (R8 sink-held K moves < 0.5 x K-only at tokens 1-2) = 'attention switch' /
  active-dormant (Guo 2410.13835) = NO to this title, and the post says the sink is still the route in that case.

## 5. Order and cut times (Copenhagen time; stop new work 05:00, PR 05:45, live 06:30)

1. Freeze (data/frozen.json: sink set, random sets) and G1, G0, committed: by 03:15.
2. Status rows bf16 then fp32, 4 copies, CPU: by 04:00.
3. Reading rows R1, R6, R2, R3, R5, R7, R4 in that order; a row not started by 04:30 is cut and listed in DEVIATIONS.
4. Vera rebuilds the headline cells with her own code from the saved per-prompt outputs, by 04:30.

## 6. Compute (memory rule)

One job at a time: the 2-block tail of Qwen3-0.6B on cached block-26 inputs, about 1.5 GB RSS (massive-dims),
CPU 2 threads or MPS. Never more than 2 jobs, never more than 5 GB. No NDIF, no API calls ($0).

## 7. Saved outputs

Per copy x precision x row (and per random set): the 200 per-prompt P(X) values in `data/patch/*.json`, so Vera and
critic2 rebuild every share from raw outputs.
