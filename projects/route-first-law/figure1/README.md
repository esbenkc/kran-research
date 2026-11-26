# Figure 1 (Owain), route-first-law

Frozen 2026-11-26 ~07:35Z, before the counted runs finished (21 of 180 runs per model on disk).

Example-pick rule (fixed before the totals exist):
- One row per harness (web canary, shell decoy key). Each row shows the ONE observation where the trap first appears,
  with both slots of the moved route item marked (BEFORE above the trap, AFTER below it). BEFORE and AFTER bytes are
  identical except that one item (Vera T3, out_t3_web.txt / out_t3_shell.txt), so one card shows both arms.
- The quoted agent step comes from the AFTER arm (the arm that separates read order from route presence):
  gemma-4-31b-it, the lowest run index whose trap outcome equals the AFTER arm's majority outcome (pooled over both
  models), the model's `thought` at the first step after that observation.
- The page/denial bytes come from that same run: web = web/site/<token>/index.html (sha256 checked against the run's
  actions[0].obs_sha256); shell = the cloud_stat observation in the run's actions.
- Bars: pooled trap rate over exposed runs per arm from analysis.py (analyse()), k of n printed; the frozen lines
  25% (BEFORE) and 75% (AFTER) are drawn as ticks.

Build: `python3 gen.py` (reads ../../kran-research/projects/route-first-law), then render.sh.

Receipt copy: gen.py runs from the room dir swarm-defense-1126-route-first-law/figure1/ (paths are relative to it).
Cold-read gate on the PNG: 16/16 (oracle FIG1 prompts via fig1gate.py, claude-haiku-4-5).
