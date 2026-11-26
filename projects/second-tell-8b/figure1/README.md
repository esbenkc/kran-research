# Figure 1 (second-tell-8b): frozen example rule

Owain (figure1), room theory-1126-oversight-gaming. Written 2026-11-26 ~16:05 CET, after the stage-1 data
(a73c277f9) and BEFORE any stage-2 (cue B) result exists (no `data/k4/` file; the k4 log showed no T_B yet).

Figure 1 shows the post's question on one real example, then the measured stage-2 switch times. The example is picked
mechanically, so it cannot be chosen for looks:

1. Seed: the MAIN seed whose paired ratio R_s,FRESH = max(T_B(MAIN, s), 8) / max(T_B(FRESH, s), 8) is the median over
   the MAIN seeds kept by `analyze.py` (`data/analysis.json`); with an even count, the lower of the two middle seeds;
   ties -> the lower seed number. If MAIN seed s has no FRESH pair (incomplete), it is not eligible.
2. Check: t = T_B(MAIN, s), the first stage-2 check where MAIN passes the switch rule (if MAIN is censored, t = the
   last check of the run).
3. Prompt: among the 64 dev prompts, the one whose MAIN drop p_off - p_B at t is closest to MAIN's dev-mean drop at t
   (a typical prompt, not the best one); ties -> the lower dev index.
4. Shown for that prompt at check t: P(praise opener) with cue B off and on, for the MAIN tail and for the FRESH tail of
   the same seed (same stage-2 prompt order, cue flags and sampler stream), read from `data/k4/main_s<s>.json` and
   `data/k4/fresh_s<s>.json` curves. No generated text: the runs score only the first answer token (exact P over the 16
   praise-opener tokens X).
5. The chart below uses every kept seed of every arm from `data/analysis.json` (`runs`, kB = 4), never a subset.

`gen.py` (this folder) asserts each step and prints the sources into the HTML comment.
