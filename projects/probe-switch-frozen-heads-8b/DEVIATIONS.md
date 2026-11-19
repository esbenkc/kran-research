# DEVIATIONS — probe-switch-frozen-heads-8b

Changes against PREREG.md (recorded commit e9e49b3ae, pushed 2026-11-18T16:26:47Z) and against the parent's code, each
with its time and what had been seen. None of these touches the FROZEN RULE block.

## Before any 8B data (written 2026-11-18 ~08:45 PST; seen: nothing from Qwen3-8B, only the Qwen3-0.6B smoke)

- D1 (order of jobs; PREREG "Budgets, early stop, order" says "Whole-graph scoring and the audits of each model
  follow its training"). Whole-graph scoring still follows each training run at once. The audits (blind head audit,
  head clamp, LoRA split, after-training map, attention rows) of the MLP-8 models move to after MLP-8 s2 is scored,
  and ALL-8 s2 moves after those audits; the ALL-8 s0 audit comes last. Run order: ALL-8 s0 -> last-MLP s0, s1 ->
  MLP-8 s0 (-> MARK-8 s0 if null) -> MLP-8 s1 -> ALL-8 s1 -> MLP-8 s2 -> audits of MLP-8 s0-s2 -> ALL-8 s2 -> budget
  extensions of null MLP-8 runs -> ALL-8 s0 audit. Why: the status needs MLP-8 s0-s2, ALL-8 s0-s1 and the untrained
  model first; with ~14-18 laptop hours of work, the frozen rule can then be read even if the clock cuts the rest
  (Tove #24219, Kasper #24225). The budget rule is unchanged: a null MLP-8 run is still extended once the same-seed
  ALL-8 T75 exists.
- D2 (the parent's pass B is not built). The parent's "recorder build" (pass B: recorders at 29 sites, a different
  compiled graph, PREREG v1.3 of the parent demoted it to a labelled column) is dropped at 8B. Every map value uses the
  eager-tail states fed by the pass-A cache, as PREREG.md says. Saves ~25 laptop minutes per job.
- D3 (resume). train.py gains a box deadline on the awake clock (macOS CLOCK_UPTIME_RAW) and a checkpoint
  (ckpt/<name>/: LoRA weights, AdamW state split in two files < 95 MB for git, every RNG state, data order, logs),
  saved every 256 updates and at the deadline; the next job resumes it. A run that ends without an early stop keeps
  its final checkpoint, so a null MLP-8 run can be extended under the PREREG budget rule. Training math is the parent's
  (same LoRA module, loss, optimiser, data order). Tested on Qwen3-0.6B: smoke_resume.sh (losses, gradient norms and
  test rows bit-identical for: uninterrupted, cut + resume, SIGKILL after a checkpoint + resume, 32-update run
  extended to 48).
- D4 (G1 in code). The parent's summary.py measured the G1 gap on the argmax switch; PREREG.md's FROZEN RULE defines
  G1 on the strict switch. summary.py now uses the strict switch (the argmax gap is reported next to it).
- D5 (fresh clone per laptop job). The Lab queue clones the branch fresh for each job, so each job rebuilds the cache
  (~25 min) and aborts when its sha256 (raw bf16 bits, both conditions, plus the answer logits) differs from the first
  job's (data/cache_sha.txt; each job's own sha in data/logs/<job>/cache_sha.txt).
- D6 (constants). Every script reads the model constants from cfgm.py (FIRST = 28, LAST = 35, probe block 18, 32 heads
  per block, hidden 4,096 at 8B); file names say tailin / mid35 instead of the parent's b20 / mid27. The last-MLP arm
  is called mlplast (parent: mlp27).

## During the run (no rule touched)

- D7 (2026-11-18 ~15:40 PST; seen: J0b cells). J0b m7ybk186x3 pushed its outputs to labq/m7ybk186x3 instead of the
  branch: run.sh's section-B gate had overwritten the tracked PREDICTIONS.md, and that unstaged change blocked the
  queue worker's rebase. Merged by hand (a18d841f6) before J1 started; the gate now writes to the box's log dir.
- D8 (2026-11-18 ~19:45 PST; seen: every status cell in J1's log). J1 u5jl3btzzt finished with exit 0 but its
  output push failed ('RPC failed; curl 55', network); its output commit stays in the kept laptop workdir. run.sh now
  cherry-picks that commit (recover_from.txt; -x keeps the original id, author time and message, Kasper #26227) at the
  start of the next box, so the next box's push carries it; if the cherry-pick fails it copies the commit's files
  (never over a finished run) and logs their sha256 against the commit's tree. Tested on a fake kept workdir
  (cherry-pick on a moved branch head; a second pass skips the already-picked commit).
