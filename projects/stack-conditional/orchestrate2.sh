#!/bin/bash
# Tail of orchestrate.sh after its restart point: main s2 eval was started by hand in the Vera slot (lane A idle), so
# this script only finishes the skeptic/research slot, releases user s1, and evaluates it. Logs to logs/orchestrate.log.
set -u
P=kran-research/projects/stack-conditional
LOG=$P/logs/orchestrate.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
t=0; until [ -e $P/lanes/vera2.done ] || [ $t -ge 720 ]; do sleep 10; t=$((t+10)); done
~/bin/room post oracle-1030-stack-conditional --as build "Iris go (orchestrate slot): run patch_1025.py now, then touch kran-research/projects/stack-conditional/lanes/iris.done." >/dev/null 2>&1
say "slot: Iris"
t=0; until [ -e $P/lanes/iris.done ] || [ $t -ge 720 ]; do sleep 10; t=$((t+10)); done
# wait for the hand-started main s2 eval so that user s1 + its eval never make 3 heavy jobs
until grep -q "EVALRUN DONE stack_s2" $P/logs/eval_stack_s2_lr0.0001_kl1.log 2>/dev/null; do sleep 10; done
say "slot over; s2 eval done"
until mkdir $P/lanes/.lock 2>/dev/null; do sleep 1; done
cat $P/lanes/B.hold >> $P/lanes/B.q; : > $P/lanes/B.hold; rmdir $P/lanes/.lock; say "released B: $(tr '\n' '|' < $P/lanes/B.q)"
until grep -q -- "lane B END rc=0 --arm user --seed 1" $P/logs/lane_B.log; do sleep 15; done
say "eval start user_s1"
GEN=0,4096,8192 bash $P/evalrun.sh user_s1_lr0.0001_kl1 >> $P/logs/evalrun.out 2>&1
say "eval end rc=$? user_s1; ALL DONE"
touch $P/lanes/A.stop $P/lanes/B.stop
