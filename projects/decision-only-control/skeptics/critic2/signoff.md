# critic2 sign-off: decision-only-control

Final cross-family review. The openai/gpt-5.5 session hit its $2.48 cap ($2.55) after its first verdict
(REJECT #21058 on wording, numbers already CONFIRMED). This sign-off is the Claude (opus) fallback.

Verdict: PASS, answer CONFIRMED (room #21122), on blog PR #204 at fca1c38.

Rebuilt with own code (`rebuild.py`, output `rebuild_output.json`, 0.8 s) from data/pool/test/cols.npz,
data/runs/*/reads/c_*.npy and data/reads/served_d010_s0_1500.npz; prompt-clustered bootstrap (200 prompts):

| cell | rebuilt | post |
|---|---|---|
| planted word s_P, d010_s0 @1500 | 0.865 [0.856, 0.874] | 87% [86, 87] |
| paired end minus start | +0.860 [+0.850, +0.868] | CI > 0 |
| probe trace s_R @1500 | -0.003 [-0.073, +0.068] | -0.3% [-7.4, 7.1] |
| onset (s_P curve) | 0.039 @500, 0.402 @600, 0.666 @700 | 600 |
| no-coupling control null086_s0 @300 | 0.009 [0.006, 0.012] | 0.9% |
| 86-point control d086_s0 @300 | 0.925 | learned |
| fresh samples (served) | 0.808 [0.777, 0.835] | 81% [78, 84] |
| trace late (1100-1500) minus early (0-500) | +0.024 [-0.001, +0.048] | 2.4 [-0.1, 4.8] |

Preview checks: 'held in all three parts, on the changed scoring and fixed thoughts' present; arXiv 2003.04560 cited,
1906.00425 absent; compiled-graph wording replaces the unmeasured 'order of sums'. Limits: changed scoring (no std
division, penalty 0.42) and fixed thought samples; n = 1 seed. Pre-merge fix asked: lab_cost_usd 0.08 -> 2.83.
