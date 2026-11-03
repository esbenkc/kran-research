# critic2 signoff — discoverable-route

PASS on the title answer. I rebuilt the headline cells from raw JSON action logs with independent code in `rebuild.py`, without importing `scorer.py` or `analysis.py`.

Evidence from `rebuild.out`:
- Headline cells: named 49/200, runbook 35/200, linked 133/150, none 144/150.
- Status: runbook minus named = -0.070, 95% CI [-0.140, 0.000]; upper bound 0.000 < 0.1025, so CONFIRMED.
- Guards: key outside route = -0.075 [-0.135, -0.015]; exposed-only = -0.069 [-0.140, +0.002], with 199/200 exposed in named and runbook. No guard fires.
- Storage-note mechanism: 17/150 opened; 1/17 used the key; same-seed no-route twins used it 12/17; non-openers matched no-route twins 133/133.
- Formula row: p_named 0.2333, p_none 0.9600, p_linked 0.8867, p_open 0.1133; predicted 0.8776; d = +0.0090, 95% CI [-0.0141, +0.0340]. The law test is unsettled because p_open < 0.15.

Post/figure check:
- Local post lines 45, 66, 72, 79, 91-93 state the same TLDR, headline counts, openers and formula numbers.
- Figure HTML states the same 25%, 18%, 89%, 96%, and 17/150 opener turn; source comment cites the raw files and summary rows.
- Deploy preview renders the title, prereg foldout, TLDR, Figure 1 and the same numeric claims.
