#!/bin/sh
# POST-STOP lane (PREREG v1.1 'Post-stop clock' + 'Gates'), serial, one heavy job at a time, watchdog --cap-gb 9.0.
# Waits until the Phase B trainer (and its watchdog) has exited, then:
#  S1 smoke on the 11-11 674 adapter, n16 x R2 = 32 rows: A/A null (--aa), plain passes 4/5 (--plain-r 2), pass 6
#     (--cue-r 1), checked row for row against the 11-11 read of the same adapter.
#  S2 smoke on the END adapter, same rows: A/A null on the status adapter.
#  GATE check_smoke.py (thresholds frozen in its docstring before any smoke ran; Kasper #19882). The branch is chosen
#     by its exit code only, and logged:
#     pass -> END read: read_tf.py R32 x 200 TEST, seed-major, plain passes on draws 0-7, pass 6 on draw 0, stop 20:50.
#     fail -> FALLBACK (PREREG, frozen): the 11-11 read.py --joint R8 on the same 200 TEST prompts.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
while pgrep -f "rl.py --arm joint" > /dev/null; do sleep 20; done
echo "trainer gone $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/smoke_674.log -- $PY read_tf.py --policy data/runs/joint_s0/adapter.pt \
    --n 16 --R 2 --aa --plain-r 2 --cue-r 1 --out data/reads/raw/smoke_674_tf
echo "end smoke_674 rc=$? $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/smoke_end.log -- $PY read_tf.py --policy data/runs/joint_s0_cont/adapter.pt \
    --n 16 --R 2 --aa --out data/reads/raw/smoke_tf
echo "end smoke_end rc=$? $(date '+%F %T')"
if /usr/bin/python3 check_smoke.py > logs/smoke_check.log 2>&1; then
    echo "GATE PASS -> branch TF end read $(date '+%F %T')"
    $PY watchdog.py --cap-gb 9.0 --log logs/end_tf.log -- $PY read_tf.py --policy data/runs/joint_s0_cont/adapter.pt \
        --R 32 --plain-r 8 --cue-r 1 --stop-at 20:50 --out data/reads/raw/end_tf
    rc=$?
    echo "end TF read rc=$rc $(date '+%F %T')"
    [ $rc -eq 0 ] && exit 0
    echo "TF read crashed -> branch FALLBACK $(date '+%F %T')"
else
    echo "GATE FAIL -> branch FALLBACK $(date '+%F %T')"
fi
$PY watchdog.py --cap-gb 9.0 --log logs/end_fallback.log -- $PY read.py --policy data/runs/joint_s0_cont/adapter.pt \
    --splits test --R 8 --k 64 --B 32 --dec-bias -2.255 --plant-tok 14656 --cue-delta 0.10 --joint \
    --out data/reads/raw/end_fallback
echo "end FALLBACK read rc=$? $(date '+%F %T')"
