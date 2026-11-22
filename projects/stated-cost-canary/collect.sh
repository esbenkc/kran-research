#!/bin/bash
# After the laptop jobs pushed (labq wait exit 0): pull their outputs and rebuild every number from the raw data.
#   bash collect.sh            # run from projects/stated-cost-canary on branch swarm/defense-1122-stated-cost-cont
# Mini-safe: analysis is pure python (no model), < 0.2 GB RSS, ~3 s.
set -eu
git pull -q --rebase --autostash origin swarm/defense-1122-stated-cost-cont
for t in gpt-oss-20b-hidden gpt-oss-20b-visible; do
  d=web/out/$t
  if [ -d "$d" ]; then
    echo "$t: $(ls $d/raw/*/[0-9][0-9][0-9].json 2>/dev/null | wc -l | tr -d ' ') runs; last log: $(tail -1 $d/job.log)"
  else
    echo "$t: no outputs yet"
  fi
done
uv run python analysis.py --tag gpt-oss-20b-hidden,gpt-oss-20b-visible
python3 research_checks/rebuild_status.py . results/rebuild_bruno.json
python3 research_checks/says_not_rebuild.py .
uv run python proof_matrix.py
[ -f figures/figure_data.py ] && uv run python figures/figure_data.py --phase counted || true
