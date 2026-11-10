# Deviations — trap-aware-canary

Each entry: what changed against PREREG v1 (frozen sections s0-s7), when, why, and which v1 rule stays a reported row.

## D1 (23:03 PST, before any counted run): block D runs after the cost cut if spend allows

s7 cut order removes block D first when the projection exceeds $4.50. The smoke projection ($4.88) triggered the cut
(D out; B and C to 30 per cell). After the cut the projection is $3.87. Block D (the `marked` minimal-pair side arm,
~$0.20) then runs last, after A, B and C, only if logged spend is <= $4.25. Reason: the $4.50 stop still holds, and D
is outside the prediction and the status. Decided from the smoke COST only; no smoke rate was read for it.
v1 rule as a reported row: if D does not run, the post says the minimal-pair arm was cut by the cost rule.

## D2 (23:04-23:06 PST, Bruno): web lane crashed at launch; rerun with fresh tokens, no data change

What happened: the first counted web launch (23:04:25 PST, code f0045e33) failed all 990 jobs within 2.2 s
(`web/raw/failed.jsonl`, created 1794294265.2-1794294267.4) with `NameError: name 'sys' is not defined`. Cause: 8
worker threads imported the provider SDKs (openai, anthropic, google-genai) at the same moment, which left a
half-initialised module; the smoke had not hit the race. Every failed run stopped AFTER the readiness HEAD and BEFORE
its first model call: the access log holds exactly 990 HEAD lines and 0 GET lines for the 990 failed tokens, and
`web/raw/cost.jsonl` gained nothing. Fix: `web/run.py main()` imports the three SDKs on the main thread before the
thread pool (committed with the counted data). Rerun: the same command, relaunched 23:06:29 PST, `logs/counted2.log`.
Every rerun builds a new site with a fresh token and fresh hrefs; 0 of the 990 failed tokens appear in
`raw/manifest.jsonl`, and the scorer counts only manifest tokens, so no failed attempt can be counted. The key lane
was not affected (0 `.error.txt` files; its runs had started and kept running). One test run (phase `tracetest`,
gpt-5.4-nano, neutral-cued, i=0, cost $0.0012) was used to rule out a code bug; it is outside every counted phase.
No v1 rule changes. Fix sha: `9522ae9e`. Independent check (Felix, from the files at 23:08): 0 of the 990 failed
tokens overlap the 213 counted manifest rows then present; 0 non-HEAD access-log lines carry a failed token.
