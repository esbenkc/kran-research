# Device check: can a mini rerun repeat a seed bit for bit? (room oracle-1118-live-kl-off-cont, Bruno)

Why: the only FLAT seed (kl0_s0) ran on the mini; both TAKE-OFF seeds (kl0_s1, kl0_s2) ran on Esben's laptop queue.
Plan (Kasper #25288, compute gate Vera #25302): rerun a seed on the mini and diff it bit for bit against the stored
run, plus a same-host control (seed 0 on the mini vs the mini's own kl0_s0). Never a status leg.

How it ran: in a scratch dir (swarm-oracle-1118-live-kl-off-cont/devchk/) holding copies of this project's .py files,
`inputs/` symlinked, the mini's py3.14 env that ran kl0_s0 (/Users/esben/work/oracle-1113-on-policy-planted/env), every
flag as kl0_s*/run.json, each job alone under watchdog.py --cap-gb 5.0. live.py there carries one scratch-only change
(live_devchk.patch): with DEVCHK_SAMPLE_ONLY=1 it stops after update 1's two sampled batches (off = no plant, on = the
planted ' We' cue), before training.

Runs (seed, mode) and what happened:
- devchk_s1, devchk_s0: --time-only (no stored-pool read 0), MPS watermark 0.40/0.30 (run_devchk.sh). Both sampled
  update 1, then hit the MPS cap (4.74 GiB) in the update-1 training forward and stopped. So a full 50-update rerun does
  not fit the 5 GB cap: the bit-check covers update 1's sampling only.
- devA_s0: --time-only, no watermark cap, sample-only (run_sample1.sh A). Peak 3.97 GB.
- devB_s0: full mode as the stored run (read 0 on the stored pool first), no cap, sample-only (run_sample1.sh B).
  Peak 4.71 GB. Its read 0 equals the stored read 0 bit for bit (cmp_B_read0_vs_stored_s0.json, 5,600 rows, 0.0).

Result (cmp_*.json; cmp1.py prints rows whose 64 sampled tokens differ and the first differing token):
- The no-plant batch (off_00001: 32 thoughts) is bit-identical in every pair: 4 mini reruns, the stored mini seed 0,
  and the stored laptop seed 1.
- The planted batch (on_00001: 32 thoughts) is NOT repeatable on the mini itself: full-mode rerun B vs the stored
  mini run differs in 2 of 32 thoughts (from token 49 and 53); B vs rerun A differs in 1 (from token 53). The two
  --time-only reruns match each other.
- The mini rerun of seed 1 vs the stored laptop seed 1 differs in 1 of 32 planted thoughts (from token 49): the same
  size as the mini-vs-mini difference.

Reading (frozen by Vera #25302 before the run: "mini-s0 also differs => the device test says nothing" bitwise):
the bitwise device test cannot be done, because the planted sampler does not repeat itself on one machine. The
cross-host difference at update 1 is no larger than the same-host one. A seed number therefore does not fix a run's
path: a rerun of seed 0 on the mini would differ from update 1 on, so seed 0's FLAT is one draw, not "the mini's
outcome". Host stays a named limit, with this evidence. [inference] The planted build runs through the compiled
sampler with a hook that keeps the bf16 activation (mt.py:1-12, 45); its kernels may not be run-to-run deterministic.
