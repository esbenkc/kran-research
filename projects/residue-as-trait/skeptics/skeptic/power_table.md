# Vera power table (own code, before smoke)

World: logit p = agent intercept (parent pooled key rates 0.188/0.275/0.025/0.487/0.075)
+ scenario main effect N(0, tau) + agent x scenario N(0, sd) + parent agent x arm deltas.
Null = all agents share one intercept. 300 sims x 300 within-scenario-x-arm agent-label shuffles.
Thresholds as frozen in PREREG s3: YES iff stat >= 0.020 and > shuffle q95; NO iff stat <= 0.005.

## Frozen status Dmin = min(D_base, D_task)  (vera_power_v21.py -> vera_power_v21.json)
| design | tau | sd0 | sd1.0 | sd1.14 | sd1.5 | sd2.5 | null sd0 | null sd1.5 |
|---|---|---|---|---|---|---|---|---|
| 8x5 | 0.7 | YES .97 | YES .91 | YES .85 | YES .72 | YES .37 / NO .49 | NO .94 | NO .94 |
| 8x5 | 1.5 | YES .41 / NO .47 | YES .41 / NO .47 | YES .43 / NO .48 | YES .46 / NO .44 | YES .27 / NO .60 | NO 1.00 | NO .99 |
| 4x10 | 0.7 | YES .88 | YES .69 | YES .62 | YES .46 | YES .24 / NO .69 | NO .96 | NO .94 |
| 4x10 | 1.5 | YES .39 / NO .51 | YES .33 | YES .33 | YES .30 | YES .13 | NO .99 | NO .96 |

At tau 0.7 this reproduces Felix's table (8x5 YES .97/.91, null NO .94).
At tau 1.5 a perfectly stable agent trait (sd 0) reads NO 47% of the time: D_task
punishes the agent's other-task rate for not knowing the held-out task's difficulty.

## Additive alternative D_add (vera_power_add.py -> vera_power_add.json), reported as a row
D_add = log-loss gain of (task rate among the other 4 agents, shifted by the agent's logit offset
vs the other 4 agents on the other tasks) over the task rate alone.
| design | tau | sd0 | sd1.0 | sd1.14 | sd1.5 | sd2.5 | null sd0 | null sd1.5 |
|---|---|---|---|---|---|---|---|---|
| 8x5 | 0.7 | YES 1.00 | .99 | .94 | .88 | .54 / NO .37 | NO .79 (YES .02) | NO .89 (YES .04) |
| 8x5 | 1.5 | YES 1.00 | .96 | .94 | .88 | .56 / NO .32 | NO .92 | NO .87 (YES .05) |

Reading rule I will apply at rebuild: if Dmin says NO while D_add says YES and the scenario
main effect is large (measured sd of scenario logits), the NO is task difficulty, not missing carry-over.

## Compute gate (plan #14122)
API-only runner, 1 job at a time, < 1 GB RSS; no local model. My sims: single process, < 100 MB, ~80 s.
PASS: <= 2 heavy jobs, << 5 GB, no pool > 2.
