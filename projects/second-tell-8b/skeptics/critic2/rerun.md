# Rerun of the critic2 receipts (Bruno, build, Claude; 2026-11-26 ~17:50 CET)

Each script was rerun with the project's `.venv/bin/python` from critic2's own working directory
(`/Users/esben/work/theory-1126-oversight-gaming`, the cwd in each script's header) on kran-research at the receipts
commit, and its stdout was compared with the stored `.out` (the stored files end with a harness `Wall time` line,
dropped for the comparison).

| script | rc | result |
|---|---|---|
| rebuild_k4_final.py | 0 | identical to rebuild_k4_final.out |
| null_ratios.py | 0 | identical to null_ratios.out |
| retention_check.py | 0 | identical to retention_check.out |
| subspace_check.py | 0 | identical on every character the stored output holds; the stored lines are cut at 766 characters by the session log (they end in `…`), the rerun prints the full lines |
| rebuild_k4_v1.py, rebuild_k4_v2.py | 1, 1 | critic2's first attempts: both glob every JSON in data/k4, including adapters_sha256.json, and stops with KeyError 'name'. critic2 fixed it in rebuild_k4_final.py (filters `_s[5-9].json`). Not evidence. |
