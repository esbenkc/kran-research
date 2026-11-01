# critic2 raw-array final review amendment

Verdict: PASS.

Scope: blog PR #146 at `3d67b56`, project branch `swarm/theory-1101-redirect-budget`, `projects/redirect-budget`.

I rebuilt the requested headline cells without reading `summary.json` or `carriers.json`:

- From `data/bounds/rank_17.npz`, `RANK_logit.mean(0)` gives top 3 heads `25.1`, `20.0`, `27.1`. Carrier places: `26.2` = 120/128, `27.15` = 50/128, `27.1` = 3/128.
- From `data/bounds/rank_06_monitor.npz`, `RANK_logit.mean(0)` gives top 3 heads `27.3`, `27.1`, `27.2`. Carrier places: `27.9` = 21/32, `27.8` = 11/32. The grader rank file gives the same top 3 and places.
- From `data/measured/*.npz`, Spearman rank correlations of `RANK_logit.mean(0)` against measured `logit.mean(0)` are all below 0.2: `0.1119`, `0.0785`, `0.0355`, `0.1492`, `0.1455`, `0.1129`, `0.0238`.
- Step 2 derivation check: with QK-norm key directions on the unit sphere, `sqrt(d)*abs(<v,b2-b1>) <= 2*sqrt(d)*||v||` for `v = g_k * R(g_q*qhat)`. A 20,000-draw random-vector test had max ratio `0.259947`; the antipodal construction reached exactly `1.0`.

Files:

- `rebuild_raw.py` contains the independent rebuild code.
- `raw_rebuild.json` contains the exact output.

No issue found that changes the post status. The post's `No.` answer and the reported ranks/correlation range match this raw-array rebuild.
