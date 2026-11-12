#!/bin/sh
# POST-STOP lane (PREREG v1.1 'Post-stop clock'), serial, one heavy job at a time, each under watchdog --cap-gb 9.0.
# Waits until the Phase B trainer (and its watchdog) has exited, then:
#  S1 smoke on the 11-11 674 adapter, n16 x R2 = 32 rows: A/A null (--aa), plain passes 4/5 (--plain-r 2) and pass 6
#     (--cue-r 1) so pass A/F/G can be checked token for token against the 11-11 read of the same adapter.
#  S2 smoke on the END adapter, same rows: A/A null on the status adapter.
#  END read: R32 x 200 TEST, seed-major, plain passes on draws 0-7, pass 6 on draw 0, hard stop 20:50 PST.
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
$PY watchdog.py --cap-gb 9.0 --log logs/end_tf.log -- $PY read_tf.py --policy data/runs/joint_s0_cont/adapter.pt \
    --R 32 --plain-r 8 --cue-r 1 --stop-at 20:50 --out data/reads/raw/end_tf
echo "end read rc=$? $(date '+%F %T')"
