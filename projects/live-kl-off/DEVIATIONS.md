# DEVIATIONS — live-kl-off

Changes after the PREREG push (f144daf65, 2026-11-17T01:00:18Z = 17:00:18 PST). None touches the FROZEN RULE block.

- D1 (17:12 PST, Bruno; before any training update). The first chain process (started 17:04:23) died with the shell
  that started it at ~17:07, right after it finished the ' Let' placebo twin (logs/ctrl_let.log: DONE 147 s). It was
  relaunched at 17:12:30 as a detached process; it kept the finished twin and continued with the pilot's placebo read
  and seed 0 (start 17:13:22). No read or training update existed before the relaunch. Cost: ~6 minutes of the clock.
- D2 (17:25 PST, Bruno; seed 0 at update ~100, no result read beyond read 100). The laptop-queue job (tree step 5)
  needs the stored test pool, whose hidden states (h_tail.npy, h_prompt.npy, ~0.8 GB) are not committed. The job
  recomputes them from the committed tokens (data/pool_cols/test/cols.npz, byte-identical to the parent's) with
  rebuild_pool.py (forward only, no sampling; Kasper #23015, Felix #23016), builds the ' Let' twin the same way, and
  runs only if the pilot's read 0 on the rebuilt pools matches the mini (s_P 0.005616 +-0.001, s_P' -0.003132
  +-0.001). Laptop seeds therefore read rows with the same tokens but hidden states recomputed on another GPU.
- D3 (18:09 PST, Bruno; seed 0 at update ~650, last read 600 flat at s_P 0.0078). Vera #23028 asked for one served
  read (served.py, 400 fresh rows, ' We' forced vs banned at thought token 0) on seed 0's adapter_1200 if seed 0 ends
  FLAT, before the positive control: without the thought penalty the model's own thoughts may drift from the stored
  pilot thoughts. PREREG allows served.py on a last adapter as a report, never a gate. The running chain could not
  take a new step, so its parent process was stopped (kill -TERM on the chain pid only; seed 0's driver and live.py
  kept running, checked with ps) and a replacement chain (chain.py with the served step and a wait on seed 0's
  driver) took over at 18:09:06. The tree, the seeds and every frozen setting are unchanged; on FLAT the positive
  control starts ~3 minutes later.
