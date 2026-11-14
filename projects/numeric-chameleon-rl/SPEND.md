# SPEND — numeric-chameleon-rl 8B replication

Slice cap at the start: $220 of Modal GPU. Since 2026-11-13 ~17:30 PST the GPU budget is $0: every Modal GPU launch
was refused (no payment method), and the replication moved to this Mac (local compute, $0). Real spend is from
`modal billing report --start 2026-11-13`.

| # | time (PST) | launch | GPU x est. hours | est. $ | real |
|---|---|---|---|---|---|
| 0 | 2026-11-13 16:50-16:56 | H100, L4, A10G launches refused: "Please add a payment method" (nothing ran) | 0 | 0 | $0 |
| 1 | 2026-11-13 16:55 | Modal CPU: download Qwen3-8B to volume nc8b-vol | 0 GPU, 0.03 h CPU | 0.01 | in $0.00 report |
| 2 | 2026-11-13 16:57-17:35 | Modal CPU: pipeline smoke on Qwen3-0.6B; stopped by hand when the plan changed (no output used) | 0 GPU, 0.6 h CPU | 0.30 | in $0.00 report |
| 3 | from 2026-11-13 17:40 | all 8B work on this Mac (MPS) | local | 0 | $0 |

Real Modal spend (billing report, 2026-11-13 17:50 PST): $0.00. No GPU time was used.

Final check (billing report, 2026-11-14 01:19 PST): $0.00. Local MPS compute for the 8B run: ~7 h of jobs between
2026-11-13 17:49 and 2026-11-14 01:17 PST (logs/run_all.steps).
