#!/bin/bash
# Run one side job in a held heavy-job slot: write maxjobs=1, wait until at most 1 heavy python job of this project
# runs (python processes only; watchdog/time wrappers are not counted), run "$@", write maxjobs=2.
cd "$(dirname "$0")/../../.." || exit 1
MJ=kran-research/projects/shared-prefix-cache/raw/orch/maxjobs
count() { ps -axo command | grep -cE '^\S*python\S* \S*shared-prefix-cache/(rl|pfx_eval|swap|pfx_stack)\.py'; }
echo 1 > $MJ
while [ "$(count)" -gt 1 ]; do sleep 5; done
echo "slot $(date +%T)"
"$@"
echo 2 > $MJ
echo "released $(date +%T)"
