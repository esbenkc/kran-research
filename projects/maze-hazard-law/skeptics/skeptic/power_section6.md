## 6. Power (Vera, before data; `skeptics/skeptic/vera_power_final.py`, output `power_final.txt`)

Simulated agents, final frozen rule (section 4: floors, v1 median-censored = UNTESTED, v2 pilot-interval 2x rule,
shape FAILS if Pearson p <= 0.025 or Weibull likelihood-ratio p <= 0.025, df = B' - 1). 600 simulated pilots per
cell; entry at call 1 (K = 80); a lost exit chance of 0.005 per page in every agent. Cells: P(HOLDS) / P(FAILS) /
P(UNTESTED) per status agent, at n = 40 and n = 60 entered maze runs.

| simulated agent | n = 40 | n = 60 | runs at the cap |
|---|---|---|---|
| constant hazard 0.005 per page | 0.02 / 0.01 / 0.98 | 0.02 / 0.01 / 0.98 | 0.44 |
| constant hazard 0.01 | 0.47 / 0.03 / 0.51 | 0.60 / 0.03 / 0.37 | 0.30 |
| constant hazard 0.02 | 0.69 / 0.03 / 0.29 | 0.81 / 0.03 / 0.16 | 0.13 |
| constant hazard 0.05 | 0.86 / 0.06 / 0.08 | 0.91 / 0.07 / 0.02 | 0.01 |
| falling hazard (Weibull k = 0.6, median ~20) | 0.12 / 0.84 / 0.04 | 0.03 / 0.96 / 0.01 | 0.13 |
| rising hazard (Weibull k = 1.6, median ~40) | 0.08 / 0.23 / 0.68 | 0.03 / 0.42 / 0.55 | 0.08 |
| burst (0.01, then 30% quit at page 40) | 0.55 / 0.12 / 0.33 | 0.62 / 0.23 / 0.16 | 0.21 |
| mixture: 30% fast leavers (0.3), 70% stayers (0.005) | 0.00 / 0.99 / 0.01 | 0.00 / 1.00 / 0.00 | 0.31 |

In plain words: a true constant hazard is called FAILS at most 7% of the time. A Yes is only likely when agents leave
at about 1 in 100 pages or faster; at 1 in 200 the median is not reached and the agent is UNTESTED. A falling hazard
or a fast-leaver/stayer mix is caught almost always. A rising hazard is usually UNTESTED (no pilot give-ups) and is
caught only 23-42% of the time. A single late burst passes as a law 55-62% of the time at this n: the first limit.
Because a two-sided status needs every TESTED agent to hold, two true-law agents at 0.02 give CONFIRMED only if both
hold; a REJECTED from a true law has probability about 1 - 0.97^2 = 0.06.
