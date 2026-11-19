#!/bin/sh
# After every run stopped: scores, the frozen-rule analysis, the proof matrix and the discarded-attempts list.
set -e
cd "$(dirname "$0")"
.venv/bin/python score.py --phase main
.venv/bin/python score.py --phase gate
.venv/bin/python analysis.py
.venv/bin/python proof_matrix.py > /dev/null
.venv/bin/python discarded.py
