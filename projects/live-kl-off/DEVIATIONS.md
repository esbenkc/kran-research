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
- D3.1 (18:09 PST, Iris for Felix #23030; written at the same minute as D3, same read, its details). The served read
  is descriptive, NOT a status leg and not in the FROZEN RULE block. It runs only if seed 0 ends without a take-off
  (FLAT or EDGE): parent served.py, 200 TEST prompts x 2 fresh draws from adapter_1200 = 400 rows, ' We' forced vs
  banned at thought token 0 with common random numbers, served s_P with a prompt-clustered interval. One heavy job
  alone (~6 GB in the parent). If seed 0 takes off, it does not run.
- D4 (18:11 PST, Iris for Felix #23037, Kasper #23036; seed 0 at update ~680, reads 0-600 seen, all s_P <= 0.0078;
  no read at 700 or later exists). Added splitter cell R ("replay"), exploratory, NOT a status leg, not in the FROZEN
  RULE block, never a line in the ONE figure (one sentence in the post + one proof-matrix row; Kasper #23036 d).
  - What: the cached engine (cached.py), thought KL off, seed 0, the d010 recipe (no-std, lam 0.42, lr 3e-4, decision
    KL 0.1, P 8 x G 8 = 64 rows per update, the live recipe's batch shape), trained on a FIXED pool built from seed 0's
    OWN fresh batches in update order: the first 143 probe-on batches (data/runs/kl0_s0/batches/on_*.npz) and the first
    143 probe-off batches = 4,576 + 4,576 rows (the cached pools had 4,584), groups of 8 thoughts per prompt as
    sampled. Tokens, the ' We' flag x1 and the probe scores M_read/M_solve (scored on the We-text) come from the batches;
    block-26 inputs come from a forward-only pass (no sampling). Reads on the same stored test pool, every 100.
  - Why: against live seed 0 the only change is reuse versus resampling (freshness). Against cached d010_s0 the change
    is the text (the sampled plant with the model's own continuation after ' We', and the probe scoring the We-text).
  - Readings, frozen now (Kasper #23036 a, stricter than #23033): s_P >= 0.30 at 2 consecutive reads by read 700 ->
    "freshness carries the brake: the same text is learnable when reused"; s_P < 0.10 at read 800 -> "the brake is in
    the text (the sampled plant or the probe reading ' We'), not freshness"; anything else, or no read 800 by the
    20:00 stop -> "not decisive". n = 1, exploratory in every case.
  - When: after the positive control (one heavy job at a time); pool build scripted and smoke-tested on 2 batches by
    19:00 (Bruno). R starts only if it can start by 19:30 PST (read 700 needs ~25 min at 2.0 s/update + reads);
    otherwise seed 1 runs per the tree. --stop-at 20:00 holds.
- D4.1 (18:12 PST, Iris from Bruno #23045; seed 0 at ~700, no read >= 700 seen by the room; cell R not built). The
  pool in D4 is changed to what cached.py can take: it needs the same prompt set in both pools (cached.py:136 assert),
  so build_replay.py keeps only prompts present in both seed-0 batch sets: the first 145 probe-on + 143 probe-off
  batches (born <= update 572) -> 271 prompts, 3,392 probed + 3,384 unprobed rows (8, 16 or 24 thoughts per prompt),
  ' We' share 0.1017 probed and 0.0038 unprobed (Bruno's numpy dry run on the raw batches). This is 26% fewer rows than
  d010's 4,584, so each row is reused ~1.35x more often. Carrying uses per update stay ~3.3 (the frequency law's
  count), but there are fewer distinct carrying thoughts (~345 vs ~458), which may make a fixed-thought onset easier,
  not harder: so the high branch ("freshness carries the brake") is the weaker of the two readings, and the post says
  so if it fires. Command: cached.py --name replay_s0 --off data/pool/replay_off --on data/pool/replay_on --test
  data/pool/test --seed 0 --steps 1500 --no-std --lam 0.42 --read-early 100 --read-every 100 --stop-at 20:00. Clock
  gates in chain.py: build only if before 19:22, training only if before 19:30; else seed 1. No smoke beside seed 0
  (Vera's one-heavy-job rule); a failed build falls back to seed 1. Readings unchanged from D4.
- D5 (18:59 PST, Iris for Selma #23074; seed 0 finished FLAT, nothing about it changes here). `oracle.py check` flags 8 files under data/
  with an mtime before the prereg push (01:00:18Z). They are INPUTS copied from the parent before the push, not results.
  Each is byte-identical to the parent (sha256 below; 'branch' = origin/swarm/oracle-1113-on-policy-planted, 'clone' =
  ~/work/oracle-1113-on-policy-planted, used for the two uncommitted hidden-state files; checked by Iris with shasum):
  - data/prompts.jsonl  bbdb848535d81c6737744a0a49a6be42f4940e89339aacc4fade1fee59e15c5c  (branch, mtime 2026-11-16T16:56:09 PST)
  - data/tokens.json  3d17f4f5fb17f59552020a3819887e2bda0d7a69545e4dd21d05b01842512604  (branch, mtime 2026-11-16T16:56:09 PST)
  - data/probe_v14.pt  a8341220f9b69b340bd67c81c7613120764eba13052e9844d3f8380124f095cb  (branch, mtime 2026-11-16T16:56:09 PST)
  - data/runs/pilot/adapter.pt  3ee161ade6afa9e46117822bc984c73a5560c45c3726b9dd88653e9f4746c5b4  (branch, mtime 2026-11-16T16:56:09 PST)
  - data/pool/test/cols.npz  f085c32fae416a0116dddbc3283e020a48139f46198730953708ceaaf81e14d7  (branch, mtime 2026-11-16T16:51:55 PST)
  - data/pool/test/meta.json  2f155f61039896b2c8fc4ac33aa93bca821532e39357495e8a93619abc0e69e3  (branch, mtime 2026-11-16T16:51:55 PST)
  - data/pool/test/h_prompt.npy  0163e505813960016710cdd8c978dbbce66831eb0e999d1b0382c733de08ee45  (clone, mtime 2026-11-16T16:51:54 PST)
  - data/pool/test/h_tail.npy  ed647cd6dd5a517338ddcfb755c72b73cef684c62684c36861797b6fa4596a06  (clone, mtime 2026-11-16T16:51:55 PST)
  No mtime was changed to pass the check. The check stays a FAIL on this line; the PR body quotes this entry.
