#!/bin/bash
# Take one heavy-job slot: write maxjobs=1, wait until <= 1 heavy job of the project runs, run "$@", write maxjobs=2.
MJ=/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/orch/maxjobs
echo 1 > $MJ
# count only the python processes themselves (watchdog/time wrappers carry the same script name after '--')
while [ "$(ps -axo command | grep -E '^\S*python\S* \S*shared-prefix-cache/(rl|pfx_eval|swap|pfx_stack)\.py' | wc -l)" -gt 1 ]; do sleep 10; done
echo "slot $(date +%T)"; ~/bin/room post oracle-1102-shared-prefix-cache --as skeptic "Vera eval START (maxjobs=1): $*" >/dev/null
"$@"
echo 2 > $MJ; ~/bin/room post oracle-1102-shared-prefix-cache --as skeptic "Vera eval STOP (maxjobs=2)." >/dev/null
echo "done $(date +%T)"
