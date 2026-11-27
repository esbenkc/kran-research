# Figure 1 (Owain), reach-law-gpt-oss

Frozen 2026-11-27 09:10 CET, before any counted run exists (labq 141r3us66d queued; data/ is empty).

Example-pick rule (fixed before the data):
- One pair of counted runs with the SAME index i (common random numbers: same ticket, bucket and decoy key) in two
  arms: NONE (no route anywhere) on the left, RUNBOOK (route in step 5 of /runbooks/ops.md) on the right. The two
  cards differ only in that step-5 clause; gen.py dies unless both runs share ticket, bucket and key.
- The right-hand outcome is set by the status from analysis.py (results/analysis.json):
  CONFIRMED -> a RUNBOOK run that was reached and did NOT use the key;
  REJECTED  -> a RUNBOOK run that was reached and DID use the key;
  UNCONFIRMED -> the majority outcome among the RUNBOOK reached runs (tie: key used).
  The left-hand outcome is the majority outcome of the NONE arm (tie: key used).
- i = the lowest counted index 0..44 where both runs exist and have those outcomes. If no index has both, the left
  and right runs each take their own lowest matching index, and the figure says "different tickets".
- The quote is the run's own `thought` (final-channel JSON) at its decision step: the first key-use action if the
  key was used, else the first size-report call, else the step after the first denial. The cards show
  ops_text step 5 and the first denial (actions[first_denial_action].observation), verbatim, cut with "…".
- Bars: pooled reached and never-reached bins (analysis.json bins), k of n, with the frozen 25% / 75% lines.
- PENDING (no data yet): i = 0, cards from shell/env.py WorkspaceEnv(0, arm) (same bytes the model will see), no
  outcomes, bars say the runs are waiting for the laptop queue.

Build: `python3 gen.py` (reads ../../kran-research/projects/reach-law-gpt-oss), then `render.sh`.

## Continuation (when ka12ju3i4s has landed)

1. `git pull` both clones. From this folder: `python3 gen.py --out <blog clone>/assets/figures/reach-law-gpt-oss-fig1`
   (reads data/gpt-oss-20b/raw through analysis.load_runs/analyse; it dies if a card fragment is not verbatim in the
   picked run's logs or if the paired runs differ in ticket, bucket or key).
2. Render from the blog clone root: `CHROME_PATH=... node _thumbs/figure1.mjs reach-law-gpt-oss-fig1`; open figure1.png.
3. text.json holds a head, subtitle and alt per status (CONFIRMED / REJECTED / UNCONFIRMED with head_key|head_route).
   Before shipping, make the subtitle's answer sentence use the TLDR's own words and numbers (no number the TLDR lacks),
   then run the oracle Figure 1 gate a few times on the post (PNG-only cold read must state question and answer).
4. Paste the new `figure1_alt.txt` (written by gen.py, double quotes already turned into single) into the post's
   front matter `figure1_alt:` in the same commit as the PNG. The pending alt in the post must not survive.
