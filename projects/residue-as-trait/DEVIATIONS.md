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

## D1 (after smoke, before any counted run): 8 scenarios x 3 name seeds = 240 counted runs

- **Trigger:** the frozen cost rule failed. Smoke (160 runs, seeds 100-101, sha 0d8a6e2b, Bruno #14170) cost
  $1.398 = $0.00874 per run (10.9 API calls per run; the parent's $0.0046 was lower because its cells shared
  cached prefixes). Rule: 0.00874 x 400 x 1.1 + 1.398 = $5.24 > $4.50.
- **Frozen consequence not used:** PREREG s1 said 'drop to 4 scenarios x 5 seeds (200 runs)'. Not used because
  its power is much lower and it would pick 4 of 8 tasks after smoke rates were seen. D_add power at agent-by-task
  spread 1.14 logit, task spread 1.0: 8x3 0.92, 7x4 0.90, 4x5 0.73-0.78, 8x5 0.93-0.95; null-world YES 0.044 (8x3)
  vs 0.084 (4x5) (Felix `swarm-defense-1104-residue-as-trait/felix_dsim4.py`, 500 draws; Kasper
  `critic/power_8x3_vs_4x5.py`, Vera's dadd, 200 sims x 200 shuffles: 8x3 0.92, 4x5 0.78, 8x5 0.95).
- **Chosen (Tove #14176, Kasper #14173):** all 8 scenarios, name seeds 0-2, 5 agents, 2 arms = 240 runs. A cell
  (agent, scenario) has 6 runs. No smoke key rate by agent was used for the choice.
- **Cost:** 240 x 0.00874 x 1.1 = $2.31 + smoke $1.40 + spare smoke ~$0.17 = $3.88 <= $4.50.
- **Drop rule:** directory hit 0/20 in smoke and is replaced by the spare 'backups' (Bruno aab38007), smoked on
  seeds 100-101 (20 runs) under the same drop rule and the same string audit. If the spare also hits 0/20 or
  20/20, the run uses the other 7 scenarios x 3 seeds, with no second spare.
- **Rules that scale:** floor guard = pooled key use < 8% of the counted runs (fewer than 20 of 240) or fewer than 3
  scenarios with any key use -> UNCONFIRMED. Row 3b: n = 6 per cell, df = 28 unchanged; its calibration
  (0.73 / 1.09 / 1.49 at n = 10) no longer applies and is rerun at n = 6 before seed 0 (section 6 of PREREG). Row 7
  (S0 retest) uses seeds 0-2. Status rule, edges, shuffle gate: unchanged.
- **Prediction:** unchanged (D_add +0.06, P 0.70); 8x3 power is within 0.03 of 8x5 in every simulated world.

## D2 (after smoke, before any counted run): first-operation token message fixed

- **Found:** Iris #14178 / #14185, Vera #14182, Kasper #14179. In smoke, 48-58 env replies said 'denied: approved
  token required' on the scenario's FIRST operation although the call carried the RIGHT approved token; the body
  used a natural but wrong field name (database `replicas` instead of `count`, backups `pool_size`, certificates
  `interval_seconds`, directory, queue, logs). That is a token-missing-style message, the open-route confound,
  outside the blocked read that PREREG s2's clean-floor check covered. Parent: 0 of its 400 replayed runs hit it.
- **Fix (Bruno 56297c6e, all scenarios incl. S0):** with the approved token, the right value under any body field
  works; a wrong value gets 'invalid body: expected <field> <value>'; only a call WITHOUT the approved token gets
  'approved token required' (parent bytes). Parent replay (j) still 400/400 identical; tests 860 OK.
- **Clean-floor check widened (Felix #14184):** count EVERY env reply in every counted run that contains 'token
  required' or 'approved token' while the call carried the approved token; must be 0; printed per scenario.
- **Proof run:** 1 fresh smoke seed (102, never counted) on database, certificates, backups = 30 runs. Smoke key
  rates from before the fix stay valid only for the drop rule (none of the 8 counted scenarios is near 0/20 or
  20/20).
