# SPEND — stack-conditional at Qwen3-8B (Modal)

Slice cap: **$220** (of the $500 shared GPU budget). Modal workspace `hi-56038`, app `stack-conditional-8b`.
Rates (from `modal billing rates`, 2026-11-13): H100 $3.95/h, CPU $0.0473/core/h, memory $0.008/GiB/h.
Every launch is `modal run` (never deploy), and every function has a `timeout=`.
Estimates are written BEFORE each launch. Real spend comes from `modal billing report`.

| # | time (PST) | launch | estimate (before) | worst case (timeouts) | real (after) |
|---|---|---|---|---|---|
| 1 | 2026-11-13 16:55 | `--phase base`: download (CPU) + 4 x cache (H100, ~10 min each) + gate (H100, ~10 min) | 5 x 0.17 h x $3.95 + CPU/mem ≈ **$3.5** | 4 x 0.75 h + 0.67 h = 3.7 h ≈ $15 | refused at launch: "Please add a payment method to use H100! GPU functions". Image built, no container ran: **$0** |
| 2 | 2026-11-13 16:57 | `--phase premise` (SC8B_GPU=L4): download Qwen3-8B (CPU) + one L4 premise check, 48 test prompts x 4 builds x 2 reads, NOT frozen data | L4 0.25 h x $0.80 + 8 CPU + 48 GiB ≈ **$0.4** | timeout 15 min (L4) + 40 min (download CPU) ≈ $0.6 | refused at launch ("add a payment method to use L4"): **$0** |
| 3 | 2026-11-13 16:58 | same premise check on A10G | ≈ $0.4 | ≈ $0.6 | refused at launch (A10G): **$0** |
| 4 | 2026-11-13 16:59 | `--phase download` (SC8B_GPU="", CPU only): Qwen3-8B weights (16 GB) to the volume | 4 CPU x 0.15 h + 8 GiB ≈ **$0.04** + volume 16 GiB x $0.09/month ≈ $1.5/month | timeout 40 min ≈ $0.2 | ran 83 s (CPU): ≈ **$0.01** (+ volume storage) |
