# Does a bound computed before training predict which attention heads a learned switch hijacks?

**Question:** Qwen3 normalises each query and key, which caps how far a change at the earlier tokens can move a
head's attention scores. Does that cap, computed from the untrained model, name the heads a learned switch later moves?
**Sources:** the saved runs of probe-switch-frozen-heads (Qwen3-1.7B, 3 runs) and sink-value (Qwen3-0.6B, 4 runs).
**Method:** carrier per run by a frozen single-head swap rule; the cap (row RANK) per head from base weights; status
by PREREG sec 7 (top-3 hits vs frozen baselines). See README.md.

## Findings

No. H = 0 of 5 runs with a clear hijacked head (2 runs had none). The 4 hijacked heads sit at places 120 and 50 of 128
(Qwen3-1.7B) and 21 and 11 of 32 (Qwen3-0.6B) under the cap. The sound rows K and C held in every head and prompt.
Spearman(cap, measured score change) per run 0.02-0.15; the measured change names the hijacked head (1st in 4 of 5).

## Blog post

Slug: `blog.kran.ai/_notes/redirect-budget.md`, figure `redirect-budget`.
