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

## D2 — exploratory row: R0 and R2 on a torch nightly that carries PyTorch PR #200559 (Iris #5177, Felix decision; no probe-vs-none comparison seen)

Basis: source facts only. PyTorch commit decc6d2308 (2026-10-12, PR https://github.com/pytorch/pytorch/pull/200559,
"[MPS] [Inductor] Compute in float32 after casting to fp16/bf16") makes Metal `to_dtype` return
`static_cast<float>(static_cast<bfloat>(x))`, i.e. round once and re-widen. It is not in torch 2.14.1 (our stack) nor
in release/2.15. If D1's reading is right, R2 on that build should round once at the split, like R2f does by hand.

Change: one exploratory job, after Bruno's planned queue and alone (same 1-heavy-job rule, ~4.6 GB): the same
`bits.py` with `--emulate 0 --fixes none,round16` under a separate venv (`nightlyenv-venv`, a clone of the run's venv
with only torch replaced by a nightly built after 2026-10-12; exact version recorded in the run's meta). R0 in the
same process is the validity check. Status and prereg clauses are unchanged and decided on torch 2.14.1 alone. If the
nightly fails to install or compile by 04:00, the row is dropped and named in Next posts.

## D3 — after results: fresh-detector seeds 1 and 2, timing of every build, head-clause diagnostic (Tove #5224, Felix #5210, Bruno #5249)

All three were added AFTER the seed-0 fresh detector (91.0% whole graph on R2), the head clause and the R2 timing were
seen. None changes the status rule, the title arm or a prereg clause; each is reported as a robustness check.
(a) Fresh detector seeds 1 and 2 on R2 (same recipe as seed 0), if they finish by 04:15 CEST. The status stays decided
by seed 0, as PREREG.md fixes ("one seed (0)").
(b) Speed of every build (R2f, opaque split, precision emulation, besides R2), one interleaved rotation
(`raw/timing/timing_all.json`). The prereg speed clause stays on R2 vs R0.
(c) Head clause: a rerun that logs the share of Yes/No answers and the probe-on vs probe-off separation per ablation,
because exactly 50.0% after removing one head can mean the detector answers the same for every input. The clause
stays scored as worded in PREREG.md; the diagnostic decides only the foldout wording.

Outcome of D2 (for the record, no decision): on torch 2.16.0.dev20261022 (git ad8c2fc) R0 still changes 64/64 and R2
changes 0/64 (A/B, passes 0-3, floor 0); `raw/bits_nightly/`.
