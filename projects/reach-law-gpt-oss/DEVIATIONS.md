# Deviations — reach-law-gpt-oss

v1.1 (Iris, 2026-11-27 08:15 CET; facts known only after the PREREG push, no rule change):
- Freeze time. PREREG.md was first pushed in commit 0e61aa72f at 07:03:52Z, but GitHub logged that push as
  activity_type 'branch_creation', and `oracle.py prereg` reads only activity_type 'push'. The next normal push
  (07:08:08Z, head d730948a8, code only: harness, figure builder, proof-matrix script; no data) contains 0e61aa72f, so
  the recorded freeze time is 07:08:08Z (08:08 CET). The frozen file is the same bytes (sha256 in the header).
- Harness commit: 05672a436 (shell/env.py WorkspaceEnv.reach(), key_calls without submit_report; test_arms.py all pass).
- Before the freeze, only parent re-counts and Qwen3-1.7B smoke runs on non-counted env seeds 800+ (smoke/) ran.
