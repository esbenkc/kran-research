# SPEND — numeric-chameleon-rl 8B replication (Modal, workspace hi-56038)

Slice cap: $220. H100 on Modal: about $3.95/h (plus small CPU/memory charges). Each launch: estimate before, real
spend after (`modal billing report`).

| # | time (PST) | launch | GPU x est. hours | est. $ | cumulative est. $ |
|---|---|---|---|---|---|
| 1 | 2026-11-13 17:20 | download Qwen3-8B to volume (CPU only) | 0 GPU, ~0.2 h CPU | 0.05 | 0.05 |
| 2 | 2026-11-13 17:25 | probe check on base model, 64 dev prompts, 16 compiles | 1 x H100 x 0.7 h | 2.80 | 2.85 |
