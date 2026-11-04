# Deviations — residue-as-trait

Every change after PREREG v1 (commit 16cd26cc), with its reason. The v1 rule stays a reported row.

## D0 (pre-data: before any smoke or counted run; no key counts of this room seen)

Status statistic changed twice, both before the harness existed:

1. v1 (16cd26cc, 01:01:55 room time) froze D_base alone (Tove #14106 A). Tove's FINAL RULING #14114 (sent
   01:00, read after the commit) replaced it with Dmin = min(D_base, D_task). v1.0.1 (fc425a38) froze Dmin.
2. Vera's power rebuild #14133 (`swarm-defense-1104-residue-as-trait/skeptic/power_table.md`) showed that Dmin reads a
   perfectly stable trait as REJECTED with P 0.47 when task base rates spread (task spread 1.5 logit), because
   D_task asks the agent's raw rate to know the held-out task's level. Tove RULING #14138 + #14145: status = D_add
   (the agent's offset added to the task rate), same edges, same shuffle gate, same floor guard. Kasper #14144 /
   #14147: use the bounded Beta(1,1) offset form, not a ridge fit; add a max-influence row. v1.0.2 freezes D_add.
3. Predicted number moved with the statistic: D_base +0.025 (v1), Dmin +0.03 P 0.55 (v1.0.1), D_add +0.06 P 0.70
   (v1.0.2, Vera #14143, Tove #14145). All three are reported next to the measured values.

D_base (v1) and Dmin (v1.0.1) are scored as reported rows with the same edges (PREREG section 4 row 1).
