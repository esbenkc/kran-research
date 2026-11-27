# Deviations — tell-chain-8b

PREREG.md pushed 2026-11-27T14:13:07Z (commit 92fa7c588, header 9c4591a17). Every entry below was written before
any file under data/ existed unless it says otherwise. None changes the frozen status rule, a threshold or the
primary metric (no RULE CHANGE).

- D0 (code after the stamp, logging only): commit f7f3b781c (pushed before the launch at 15:14 CET) adds one line to
  tc.py (`mem(f"g {g}")`: MPS driver/current memory next to the every-64-update timing line) and a side logger to
  run_job.sh (data/mini/mem.log every 120 s: memory_pressure free % and the python process footprint). This was
  Vera's compute-gate condition (#32083). No training, evaluation or scheduling line changed
  (`git diff 92fa7c588 f7f3b781c -- projects/tell-chain-8b`).
- D1 (smoke size, disclosed): PREREG says the memory smoke runs "at most 16 updates on TRAIN prompts". The smoke
  (smoke/smoke1.log, TT_ROOT=smoke) ran every record for at most 16 of its own updates (stage-len 16, cap 16), but it
  chained three stages, so the batch ran 48 global updates; its dev evaluations read the dev prompts, as the parent's
  smoke did. Nothing from smoke/ is used anywhere.
- D2 (memory, reported after the run; no effect on any number): the real job peaked at 9.23 GB footprint
  (data/mini/tc.log, `/usr/bin/time -l`: 9228814496 peak memory footprint), 0.23 GB over the 9.0 GB solo line; the
  smoke at the same 20 slots measured 8.10 GB (smoke/smoke1.log). System free memory stayed at or above 22%
  (data/mini/mem.log, memory_pressure every 120 s); mem-guard did not act. The run finished in one process, no
  restart, no checkpoint resume (tc.log has one `inits` line and no `RESUME`).
- D3 (logging only): the run_job.sh wrapper shell did not write its END line to data/mini/chain.log (the shell exited
  before the python job; the orphaned memory-logger subshell was stopped by hand at 07:33 PST after the job ended).
  The job's own end markers are in data/mini/tc.log: `DONE g 1504 train_s 4778.6` and the `/usr/bin/time -l` block.
- D4 (post-run export, no new training): base_px.py reran tc.py's setup evaluation (LoRA off, base b) after the job and
  wrote the untrained tail's per-prompt dev lists per seed and cue to data/base_px.json for the skeptics' resampling;
  its gap means equal run_meta.json gap_base exactly (max abs difference 0.0).
