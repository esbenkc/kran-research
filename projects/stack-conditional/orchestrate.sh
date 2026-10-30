#!/bin/bash
# Tonight's second half, sequenced so that at most 2 heavy jobs run (one eval at a time, lanes held while it runs).
# Started once from ~/work/oracle-1030-stack-conditional; every step logs to logs/orchestrate.log.
set -u
P=kran-research/projects/stack-conditional
LOG=$P/logs/orchestrate.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
waitfor() { until grep -q -- "$1" $P/logs/lane_$2.log; do sleep 15; done; say "seen: $1"; }
release() {  # move lanes/<L>.hold to the end of lanes/<L>.q
  until mkdir $P/lanes/.lock 2>/dev/null; do sleep 1; done
  cat $P/lanes/$1.hold >> $P/lanes/$1.q; : > $P/lanes/$1.hold
  rmdir $P/lanes/.lock; say "released $1: $(cat $P/lanes/$1.q | tr '\n' '|')"
}
ev() { say "eval start $*"; GEN=${GEN:-8192} bash $P/evalrun.sh "$@" >> $P/logs/evalrun.out 2>&1; say "eval end rc=$? $*"; }

if [ -z "${RESUME:-}" ]; then
# 0. shuffled into lane B once Vera's slot is over (flag file) or at 12:15 PDT (20:15 CET)
until [ -e $P/lanes/vera.done ] || [ "$(date +%H%M)" -ge 1215 ]; do sleep 15; done
until mkdir $P/lanes/.lock 2>/dev/null; do sleep 1; done
head -n 1 $P/lanes/B.hold >> $P/lanes/B.q; tail -n +2 $P/lanes/B.hold > $P/lanes/B.hold.tmp; mv $P/lanes/B.hold.tmp $P/lanes/B.hold
rmdir $P/lanes/.lock; say "shuffled released to B"
# a. always-penalised eval in lane A's slot, then main s2
waitfor "lane A END rc=0 --arm user --seed 0" A
ev penall_s0_lr0.0001_kl1
release A
fi
# b. shuffled + user s0 evals in lane B's slot, then user s1
waitfor "lane B END rc=0 --arm shuf --seed 0" B
ev shuf_s0_lr0.0001_kl1
GEN=0,4096,8192 ev user_s0_lr0.0001_kl1
# b2. skeptic/research slot (one job at a time, lane A runs main s2 beside it): Vera, then Iris; each touches a flag
~/bin/room post oracle-1030-stack-conditional --as build "Vera go 2 (orchestrate.sh slot): run vera_patch.py now, then touch kran-research/projects/stack-conditional/lanes/vera2.done. @Iris you go after Vera's flag; touch lanes/iris.done when finished. Timeout 12 min each." >/dev/null 2>&1
say "slot: Vera"
t=0; until [ -e $P/lanes/vera2.done ] || [ $t -ge 720 ]; do sleep 10; t=$((t+10)); done
~/bin/room post oracle-1030-stack-conditional --as build "Iris go (orchestrate.sh slot): run patch_1025.py now, then touch kran-research/projects/stack-conditional/lanes/iris.done." >/dev/null 2>&1
say "slot: Iris"
t=0; until [ -e $P/lanes/iris.done ] || [ $t -ge 720 ]; do sleep 10; t=$((t+10)); done
say "slot over"
release B
# c. main s2 eval (lane B runs user s1 beside it)
waitfor "lane A END rc=0 --arm stack --seed 2" A
GEN=0,4096,8192 ev stack_s2_lr0.0001_kl1
# d. user s1 eval
waitfor "lane B END rc=0 --arm user --seed 1" B
GEN=0,4096,8192 ev user_s1_lr0.0001_kl1
say "ALL DONE"
touch $P/lanes/A.stop $P/lanes/B.stop
