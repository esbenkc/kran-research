#!/bin/bash
# Vera: act-level rebuild, serial (one job at a time). Run only when the heavy slot is free.
set -e
cd "$(dirname "$0")"
PY=/Users/esben/work/theory-1020-megaprobe-after/kran-research/.venv/bin/python
A="a__gap__lam10000__seed0 a__gap__lam10000__seed1 a__gap__lam10000__seed2"
B="b__gap300__lam10000__seed0 b__gap300__lam10000__seed1 b__gap300__lam10000__seed2"
$PY vera_feats.py $A $B ${EXTRA}
$PY vera_scorecheck.py b300 $B > vera_scorecheck_b300.out 2>&1
$PY vera_know.py clean $A $B > vera_know.out 2>&1
$PY vera_klnpy.py $A $B > vera_klnpy.out 2>&1
$PY vera_panel.py a $A > vera_panel_a.out 2>&1
$PY vera_panel.py b300 $B > vera_panel_b300.out 2>&1
$PY vera_refit.py a $A > vera_refit_a.out 2>&1
$PY vera_refit.py b300 $B > vera_refit_b300.out 2>&1
echo ALL DONE
