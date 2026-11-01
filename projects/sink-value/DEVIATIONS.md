# DEVIATIONS — sink-value

Changes after PREREG v1 (8e73c6a2). Each entry: id, who decided (room post), what had been seen. Git commit time is the
record. PREREG v1.1 (86e05125) and v1.2 (3bab74e6) were set before any freeze output or patch row; they are listed in
PREREG.md itself.

## D1 (Bruno #11226, Felix; seen: the G1/G0 gate on grader s0 CPU bf16 only): CPU threads 6 instead of 2

PREREG s4 names CPU (2 threads) as the status device, as in massive-dims. On grader s0 CPU bf16 the per-prompt P(X)
for on, off and G0 are bit-identical with 2 and 6 threads (`data/gate/*threads6.json`), and bit-identical to
massive-dims `ref/grader_s0_bf16_cpu_status.json` (max diff 0.0). The status chain therefore runs 6 threads (3.4 s per
pass instead of 6.6 s). No number, set, share or status rule changes. The gate for every other copy and precision
still compares against massive-dims row (d) within 0.005 before its status rows count.
