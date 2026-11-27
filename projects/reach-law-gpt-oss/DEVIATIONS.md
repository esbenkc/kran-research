# Deviations — reach-law-gpt-oss

v1.1 (Iris, 2026-11-27 08:15 CET; facts known only after the PREREG push, no rule change):
- Freeze time. PREREG.md was first pushed in commit 0e61aa72f at 07:03:52Z, but GitHub logged that push as
  activity_type 'branch_creation', and `oracle.py prereg` reads only activity_type 'push'. The next normal push
  (07:08:08Z, head d730948a8, code only: harness, figure builder, proof-matrix script; no data) contains 0e61aa72f, so
  the recorded freeze time is 07:08:08Z (08:08 CET). The frozen file is the same bytes (sha256 in the header).
- Harness commit: 05672a436 (shell/env.py WorkspaceEnv.reach(), key_calls without submit_report; test_arms.py all pass).
- Before the freeze, only parent re-counts and Qwen3-1.7B smoke runs on non-counted env seeds 800+ (smoke/) ran.
- Laptop job (Bruno, 08:20 CET): labq 141r3us66d at commit 32277a868, `bash labq_job.sh 2026-11-27T11:45:00+01:00`
  (test_arms.py, then run.py --model gpt-oss-20b --phase all), --mem-gb 16, --est-hours 2.0, output
  projects/reach-law-gpt-oss/data. Smoke (smoke/qwen3-1.7b/, 4 runs, env seed 800, one per arm): the harness ran end to
  end; Qwen3-1.7B never opened ops.md (20 steps each), so the smoke shows plumbing only. Its first 3 runs used the
  pre-05672a436 reach() (no key_calls/report_mention fields); counted runs use the committed code.
