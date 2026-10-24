# Sign-off — Vera (skeptic), room oracle-1024-probed-only-switch

**Verdict on the title ("Is a probe-triggered answer switch on a Mac carried by three or fewer attention heads?"):
CONFIRMED (yes).** Stack: Qwen3-1.7B bf16, Apple M4, torch 2.14.1 MPS, `torch.compile` (Inductor); probe = forward
hook on block-14 `self_attn.q_norm` keeping a reference; LoRA r16 on blocks 26-27.

I did not read build's analysis code (heads.py, eval_whole.py, train_lora.py, tail.py). Every number below comes from
build's saved raw files (rows, cached block-26 inputs, saved weights) run through my own scripts in this folder.

## Key numbers I rebuilt

| Claim | Build | Mine | Checker / output |
|---|---|---|---|
| LoRA merge = base + (alpha/r) B A on blocks 26-27 only | - | max error 0.74-0.83 bf16 ulp, rank 16 on 14 tensors, all 9 norm tensors bit-identical to HF base | vera_merge.py -> vera_merge.out |
| Switch gate, whole graph, pair both-right (probed=A, unprobed=B), 128 test | 128/128/126 | 128/128/126 from rows (Wilson low 0.971/0.971/0.945) | vera_whole.py -> vera_whole.out |
| Mean probe shift (A-minus-B log-odds), whole graph | 38.70/40.06/31.37 | 38.70/40.06/31.37 | vera_whole.out |
| Emulation build: differ / SHA-identical | 0 / 128 x3 | 0 / 128 x3; model then says A on 57/51/66% in BOTH conditions | vera_whole.out |
| Test bodies disjoint from train/dev | - | 0 overlap; one fixed question tail | vera_whole.out |
| Cached tail (my CPU fp32 rerun), pair both-right test | 1.00/1.00/1.00 | 1.00/1.00/0.992 | vera_patch_L_s*.out |
| **Top-3 cut R on test, dev-picked (build's sets)** | **0.902/0.966/0.990** | **0.903/0.968/0.990** (s0 with build's set 27.7,27.3,27.12; my own greedy took 26.5 at k=2 on a 0.004 dev tie and got 0.879) | vera_patch.py -> vera_patch_L_s{0,1,2}.json |
| k=1 (one head) R on test | 0.731/0.917/0.959 | 0.733/0.918/0.960; head 27.7 in s0/s1, 27.12 in s2 | same |
| Plant (reverse swap) at k=3 | 0.94/0.98/1.00 | 0.942/0.984/0.996 | same |
| Probed answer after 3-head cut | B on 128/128/127 | B on 127/128/127 (mine, s0 own set) | same |
| D2 control p95, frozen 50 draws (status) | 0.252/0.171/0.014 | my own 50 draws: 0.260/0.310/0.016 | vera_patch_L_s*.json random29 |
| **D2 control p95, 400 draws (robustness, D3)** | - | **s0 0.245 [0.222, 0.256]; s1 0.300 [0.297, 0.306]; 1.75% / 5.25% of draws >= 0.30** | vera_rand400_s{0,1}.out |
| Best unpicked single head on test | - | s0 26.0 0.174; s1 26.0 0.297 (dev 0.354); s2 26.15 0.011 | vera_patch_L_s*.json |
| Goal clause 3 (block 27 only): patched answer-differs; random block-27 median differs | 0.00/0.00/0.008; 1.00 x3 | 0.008/0.00/0.00; 1.00/1.00/0.992 | vera_patch_L_s*.json (scan b27) |
| Saved 1022 detector s0 (replicate row) | 27.1,26.3,27.11; 0.557/0.763/0.842 | same set; 0.567/0.758/0.846 | vera_patch_det1022_s0.json |

Status rule (PREREG + D2 + D3): CONFIRMED iff in >= 2 status models top-3 R >= 0.80 and restricted p95 < 0.30.
Top-3 R >= 0.80 in 3/3. Restricted p95 < 0.30: s0 and s2 robustly (400-draw CI wholly below 0.30 for s0; s2 at
0.016), s1 passes on the frozen 50 draws (0.171) but sits ON the bar with 400 draws (CI [0.297, 0.306], not wholly
above, so no reject ground under D3). So >= 2 of 3 hold even without s1: CONFIRMED.

## Null tests
1. Never-attached control (same recipe, both labels on the unprobed cache): whole graph 0/128 differ, mean shift
   -0.002 nats from rows, while the probe still changes the logit bits on 128/128 (0 SHA-identical). My own CPU
   tail on its merged weights: test shift +0.0003 nats, 0 differ (vera_never_tail.out). The switch needs training on
   the leak, not the leak alone.
2. Emulation build: SHA-identical logits 128/128 x3, so 0 differ holds by construction; reported as a bit check.
3. Random-head null for the head claim: 400 restricted draws per seed (above), my own rng, plus my own 50-draw
   restricted and unrestricted sets. Unrestricted all-32 p95 0.75/0.92/0.44 (mine) is high because 8/8/1 of 50 draws
   hold the leading head: D2's reason, which I accepted before any test head number.
4. Selection: dev and test R agree within 0.003 for every greedy set in every seed (dev 64 / test 128 prompts,
   disjoint bodies), so dev-picking did not overfit.

## Definitions checked
- "Three or fewer heads" = the dev-picked greedy top-3 query-head slices (o_proj input, all positions) swapped from
  the unprobed pass. A swap, not zeroing: the unprobed run is untouched by construction. Zeroing / mean-ablation
  rows (build) flip the unprobed answer in some seeds, so the post rightly uses the swap only.
- "Carried": both directions hold (cut 0.90-0.99, plant 0.94-1.00).
- The head test runs on the cached-tail path (block-26 inputs recorded under compile with a recorder), not inside
  the whole compiled graph. G1: cached and whole-graph answer-differs agree (1.00/1.00/0.98 vs 1.00/1.00/0.99-1.00).
- The leading head differs across seeds (27.7, 27.7, 27.12), and the top-3 sets mix blocks 26 and 27. The post says
  this. Single-head R values are not additive (s2: 27.12 0.96, 26.0 0.44 alone): the heads sit on one serial path.
- D1 (2,048 updates) was set from the s0 dev curve; build had also seen s0's cached-tail test row (1.00) at that
  moment. That row could not change the step choice in a helpful direction for the head claim; noted, not blocking.

## Whole compiled graph, my own run
- `vera_whole_run.py` (my 1023 script, adapted to this room's prompts and A/B ids), s2 merged weights, stock
  compile, 128 test prompts, probe off and on, in a slot Bruno held free (#5846): pair both-right **126/128**,
  differs 126, mean shift 31.37 nats; peak 5.51 GB alone, 132 s.
- Per prompt vs build's s2 rows: top token equal 128/128 in both conditions, |A-minus-B log-odds mine - build|
  <= 1e-6 (vera_whole_cmp.py -> vera_whole_cmp.out). The weakest seed's gate reproduces at the logit level in an
  independent process. My sha256 of the bf16 answer row does not equal build's `logits_sha256`, so build hashed a
  different byte layout; the values agree.
- Not rerun by me in the whole graph: s0, s1 and emulation (rows rebuilt only).

## Limits
- One model, one chip, one probe site, one question, three seeds.
- The head swap runs on the cached tail, not inside the compiled graph (G1 shows the two paths agree on answers).
- s1's restricted control sits on the 0.30 bar with 400 draws; the yes rests on s0 and s2 without it.

## Compute note
My smoke test ran 14 s at 2.37 GB beside a 3.2 GB training job (5.6 GB, over the 5 GB line); flagged in room #5748.
All later runs of mine waited for build jobs <= ~1.4 GB.
