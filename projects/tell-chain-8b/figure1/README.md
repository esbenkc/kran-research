# Figure 1 (tell-chain): frozen example rule

Owain (figure1), room theory-1127-tell-chain. Written 2026-11-27 ~15:05 CET, before any stage-C or stage-D result
exists.

Figure 1 shows the post's question on one real seed's chain of cues (A -> B -> C -> D), then that seed's updates to
learn each new cue next to its same-update coin-flip control and the two pre-registered predictions (multiplying
0.74^k vs flat 0.74). The example is picked mechanically, so it cannot be chosen for looks:

1. Seed: the MAIN seed whose paired stage-D ratio vs its coin-flip control, R_D,s = max(T_D(MAIN,s),8) /
   max(T_D(CTRL-D,s),8), is the median over the seeds `analyze.py` keeps; with an even count, the lower of the two
   middle seeds; ties -> the lower seed number. A seed without a CTRL-D pair is not eligible. Before stage D exists,
   the same rule on stage C picks a provisional seed (pending renders only).
2. Stage numbers shown for that seed: T_B from the parent (second-tell-8b data/analysis.json, main and coin, k4),
   T_C and T_D from this run (MAIN vs CTRL-C / CTRL-D), exactly as `analyze.py` reports them.
3. Prompt card: at t = T_D(MAIN, s), among the 64 dev prompts, the one whose MAIN drop p_off - p_D is closest to MAIN's
   dev-mean drop at t (a typical prompt, not the best one); ties -> the lower dev index. Shown: P(praise opener) with
   cue D off and on for MAIN and for CTRL-D of the same seed at the same check. No generated text: the runs score
   only the first answer token.
4. Any group-level number (the seed-level mean and interval) comes from `analyze.py`'s output, never from this figure's
   own arithmetic.

`gen.py` (this folder) asserts each step and prints the sources into the HTML comment.
