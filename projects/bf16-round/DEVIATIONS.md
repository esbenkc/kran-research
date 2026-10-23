# DEVIATIONS — bf16-round

Changes to `PREREG.md` (frozen 2026-10-23 01:58:47 CEST, commit 7b00324). Each entry's commit time on branch
swarm/oracle-1023-bf16-round is its record. Times are Copenhagen time.

## D1 — exploratory row R2f "round16 + fp32 rope" (Iris #5162, Felix decision; no probe-vs-none comparison seen)

Basis: the Metal gate only (kernel sources from `bits.py --code-only`, 1 prompt, no saved logits). The title arm R2
passes the gate: `raw/codegate/Qwen3-1.7B/compile_emu0/code/round16_noprobe_A.py:867` has
`tmp27 = static_cast<bfloat>(tmp26)`, the rounded q_norm weight product. Iris read the same file: the Metal temporary
then has type `bfloat`, and the bf16 cos/sin make the no-probe rope run in bf16 arithmetic (:873, :876, :877), while
the probed rope kernel loads the stored q and computes in float (`round16_probe_A.py:842`). So R2 may still differ for
a reason the prereg did not foresee.

Change: add ONE exploratory row, R2f: the R2 q_norm, then `q = q.float()` before rope and `q = q.to(bf16)` after rope
(attention forward patched; same in both graphs). It tests whether ONE rounding at the split, with the rest of the
query path as in the probed build, is enough. R2f has the same measurements as R2 (bits, detector graphs if time,
speed). It does NOT change the title arm, the status rule or the prereg clauses: status is decided by R2 alone.
R2f is reported as exploratory in the post.

Also from the gate (reported, no change): `cast` and `bitcast` no-probe sources equal stock `none` apart from
comments (erased, as Iris predicted from `pointless_convert` and `remove_redundant_views`); the block-14 q_norm
reduction is the same code with and without the probe.
