# DEVIATIONS — shared-prefix-cache

Changes against `PREREG.md` (v1 @ 397783f8). Each entry: what changed, who decided (room post), what had been seen.

## D1 — C4 sanity row: two read-outs, chunked is the frozen score (Bruno #12039, Kasper #12043)

PREREG s8 scores C4 in the "whole chunked graph". The saved stack-conditional switches were trained on WHOLE 34-token
graphs, and they treat the chunked eager prefix (3-token prefill; 77% of its block-26 elements at 0-2 differ from the
whole-pass bits, `data/g0.json` pair `grading_shared_vs_stock`) as 'not grading': grader s0 flattery on chunked eager
0.379 vs 0.121 on the whole eager graph. So we report two read-outs. Frozen C4 = chunked (grading = chunked eager,
serving = compiled with its own cache, pinned = compiled with the chunked eager cache): removed 0.935 / 0.967 / 0.712
/ 0.942, 3 of 4 >= 0.8, holds. Extra row = whole (stock whole graphs; pinned = compiled rest reading the prefix K/V of
one whole eager pass, `pfx_eval.py --prefix-build eager_whole`): removed 0.989 / 0.985 / 0.985 / 0.997, 4 of 4.
Seen when decided: both read-outs (C4 holds in both, so the choice does not move the outcome).

## D2 — RL concurrency (Bruno #12050)

The orchestrator (`orch.py`) started with 1 lane at 17:12 PST because Vera's C4 rebuild (~3.7 GB) was running; lanes
follow `raw/orch/maxjobs` (PREREG s9: at most 2 heavy jobs including skeptic jobs). No change to any arm.
