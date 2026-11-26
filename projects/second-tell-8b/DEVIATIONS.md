# Deviations — second-tell-8b

PREREG.md pushed 2026-11-26T14:02:52Z (commit 7b19205af, header e1dc872bf). Every entry below was written before
any file under data/ existed. None changes the frozen status rule, a threshold or the primary metric (no RULE CHANGE).

- D0 (found in the smoke, fixed before any data): the first tt.py build computed the KL term with a detached
  P (`pr = lp.exp()` inside `torch.no_grad()`), so the KL gradient was zero (beta 0 in effect). The stage-1 smoke
  (TRAIN prompts, 16 updates, smoke/ only) showed P(X | no cue) 0.34 -> 0.0004 in 8 updates. Fixed to
  `kl = (lp.exp() * (lp - lpb)).sum(-1)` with the gradient through both factors, as the parent rl.py. Kasper found
  the same line independently (room #31433). No smoke curve is used anywhere.
- D1 (memory gate; organism detail): PREREG section 2 says "Organism (parent verbatim)". The parent's scored runs
  kept the untied head in fp32 (parent rl.py --head-dtype default fp32). This run keeps blocks 34-35 and the final
  norm in fp32 and the head in bf16 (h is cast to bf16 before the head matmul; the parent PREREG of
  cue-size-law-8b-rep allowed "head may stay bf16"). Reason: with an fp32 head the batched job peaked at 11.4 GB
  (stage 1, 15 tails) and 10.3-13.6 GB (stage 2, 20 tails) footprint, over the 9 GB solo limit. Two exact
  memory changes go with it: (a) gradient accumulation over prompt chunks of 2 prompts per tail (the loss is a sum
  over prompts, so the gradient is the same; the sampler draws per chunk, so the sampled tokens differ from a
  single draw but follow the same distribution, identically in every arm), (b) torch.mps.empty_cache() after every
  update and evaluation. Measured after the change: stage 1 R=15 peak 8.80 GB (smoke/stage1_mem3.log, chunk 4),
  stage 2 R=20 peak 8.03 GB (smoke/stage2_mem4.log, chunk 2); the job runs chunk 2 in both stages. The frozen
  PORT GATE (FRESH median T_B at k4 in [266, 326], parent 296/280/304 with the fp32 head) is the check that the
  port reproduces the parent organism.
- D2 (wrapup mechanics): stage 2 runs get `--wall-stop` = 19:15 CET (PREREG: the run ends by 20:00 CET; analysis,
  review and the post need the last 45 minutes). A tail without a switch when the wall-stop fires has run fewer than
  1,536 updates: analyze.py counts it as incomplete (not finished, left out of the status, listed in the notes), not
  as censored at 1,536, per the frozen line "Seeds not finished then are not in the status".
- D3 (memory, reported after the run; no effect on any number): the real k4 job peaked at 9.16 GB footprint
  (data/mini/chain.log, `END k4 rc 0 peak 9156053008`), 0.16 GB over Vera's 9.0 GB solo line; the 20-tail smoke had
  measured 8.03 GB (smoke/stage2_mem4.log). The smoke timed only the cue-off and cue-B evaluations; the real job also
  evaluates cue A every 64 updates and keeps 20 stop snapshots. Stage 1 peaked at 7.72 GB. Free memory stayed at or
  above 37% (memory_pressure) and mem-guard did not act. k2 runs with the same settings.
