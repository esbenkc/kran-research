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

v1.2 (Bruno, 2026-11-27 09:17 CET; pre-data design change, no counted run exists; outside the FROZEN RULE block, so
not a rule change; Tove #32007, Felix #32008):
- Stop time. PREREG.md:111 put the 11:45 CET stop on the whole job. The laptop worker has been offline since 06:52Z,
  so a start after 11:45 would have ended with zero runs. Now the stop applies to the LINKED filler only
  (run.py --filler-stop-at), and it is ignored when the job starts after that time: gate, the 135 primary runs and the
  45 PROMPT runs always run; a late (pending-laptop) start runs the whole plan, filler included. Job 141r3us66d
  (never started) was cancelled; the new job id follows below.
