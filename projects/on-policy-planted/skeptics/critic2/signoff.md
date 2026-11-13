# critic2 sign-off — oracle-1113-on-policy-planted (final cross-family review)

Session 1 ran on openai/gpt-5.5 (REJECT #21541: numbers matched; lab_cost_usd and agenda T3 stale).
It hit its $2.49 cap ($2.53) during the re-review; this re-review ran on the Claude fallback (opus), as the swarm rules allow.

## Answer to the question: NO (REJECTED) — signed
Rebuilt with own code (`rebuild.py`: Poisson(1) prompt-cluster bootstrap, 4,000 draws, own seed) from raw
`data/runs/*/reads/c_*.npy` + `data/pool/test/cols.npz`; output `rebuild_stdout.txt`, `rebuild_out.json` (0.2 s, 49 MB RSS).

| cell | rebuild | post / summary |
|---|---|---|
| live s_P(1,200) | +0.0113 [+0.0089, +0.0137] | 1.1% (0.9-1.4) |
| live onset (PREREG v1 rule, 13 reads) | none, censored at 1,200 | no take-off by 1,200 |
| live s_P range over 13 reads | 0.0056-0.0113 | 0.6%-1.1% |
| live s_R(1,200) | -0.0266 [-0.0740, +0.0248] | -2.7% (-7.6 to 2.5) |
| cached onset | (500, 600] | 600 |
| cached s_P(1,200) | +0.8951 | 90% |
| cell A s_P(800) / (876) | +0.1484 / +0.2235 | 15% / 22% |
| cached s_P(800) | +0.7256 | 73% |
| fresh1200 s_R / D_R / s_P | -0.0543 [-0.161, +0.047] / 0.1014 / +0.0139 | -0.054 (-0.160 to 0.049) / 0.1014 / 0.014 |
| carrying rate (probed, live batches) | 0.1013 (n 9,728); unprobed 0.0056 | 10.1 in 100 |

Prereg: clause 1 FAILED (no onset by 1,200, 2x the fixed onset 600); clause 2 held (s_R CI-high 0.025 < 0.10);
clause 3 held (0.1013 in [0.05, 0.15]). Status rule (PREREG v2) "no onset by 1,200" -> NO.

## Preview (deploy-preview-206 at c4e8595, modified 12:03:13Z) — fixes required in the merge commit
1. Served read: post says 0.8% (95% interval -6.3% to 7.6%); `data/reads/served_live_1200.json` s_P_ci = [-0.0641, 0.0728] -> "-6.4% to 7.3%".
2. lab_cost_usd: 1.69 in front matter; `room spend` API total $2.77 after critic2 stopped + $0.02 question choice = 2.79.
Agenda T3 (agenda-prosaic.md:57-62) updated after run: OK. Receipt links 200; /lab card present; prereg record 200.
