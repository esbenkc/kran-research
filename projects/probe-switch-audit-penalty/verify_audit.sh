#!/bin/bash
# Iris: one serial slot (~3 min, footprint ~3.4 GB, nothing else heavy beside it).
# (1) audit.py fast path vs the full 2-block pass on warm s0 (16 dev / 16 test) -> logs/verify_audit.log
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1026-audit-penalty/env/.venv/bin/python
L=logs/verify_audit.log
for mode in slow fast; do
  flag=""; [ $mode = slow ] && flag="--slow"
  /usr/bin/time -l $PY audit.py --run warm_s0 --n-dev 16 --n-test 16 --kmax 3 --n-random 2 $flag \
    --out data/audit_verify/$mode >> $L 2>&1
done
$PY - >> $L 2>&1 <<'EOF'
import json
a = json.load(open("data/audit_verify/slow/warm_s0.json")); b = json.load(open("data/audit_verify/fast/warm_s0.json"))
worst, n = 0.0, 0
same_rev = a["reverse_greedy_dev"][-1]["set"] == b["reverse_greedy_dev"][-1]["set"]
for sec in ("test_cut", "test_induce"):
    for k in a[sec]:
        if sec == "test_cut" and a["test_cut_sets"][k] != b["test_cut_sets"][k]:
            continue
        if sec == "test_induce" and not same_rev and not k.startswith(("all_32", "identity")):
            continue
        n += 1
        for m in ("removed", "recovered", "blind_removed", "switch_after"):
            if m in a[sec][k] and a[sec][k][m] is not None:
                d = abs(a[sec][k][m] - b[sec][k][m]); worst = max(worst, d)
for g in ("blind_greedy_dev", "informed_greedy_dev", "reverse_greedy_dev", "moves_greedy_dev"):
    print(g, "slow", a[g][-1]["set"], "fast", b[g][-1]["set"])
print("VERIFY sets compared", n, "worst abs diff", worst, "OK" if worst < 0.005 else "MISMATCH", "secs slow", a["secs"], "fast", b["secs"])
EOF
echo "VERIFY_DONE $(date)" >> $L
# (3) Iris research check (Tove #7229): can ONE position's block-26 input tell probed from unprobed? CPU, sklearn.
/usr/bin/time -l $PY /Users/esben/work/oracle-1026-audit-penalty/swarm-oracle-1026-audit-penalty/research_checks/sep.py \
  raw/step2/Qwen3-1.7B/compile_C data/prompts.jsonl \
  /Users/esben/work/oracle-1026-audit-penalty/swarm-oracle-1026-audit-penalty/research_checks/sep.json >> logs/verify_sep.log 2>&1
echo "SEP_DONE $(date)" >> logs/verify_sep.log
