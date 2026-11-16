# DEVIATIONS: cue-size-law-8b

Changes against PREREG.md (commit e33506dc, pushed 2026-11-16T14:03:31Z). Each entry: what changed, who decided, what
had been seen when it was decided. A line that starts with "RULE CHANGE:" marks a change inside the FROZEN RULE block.

## D0: S7 secondary read added to analyze_8b.py (Felix, 06:04 PST; Kasper #22643)

analyze_8b.py gains S7: an interpolated-crossing T (the linear crossing of D = 0.10 between the eval before the dense T
and the dense T), with per-seed slopes and a seed-level interval, reported next to the dense T. It is a secondary read
outside the FROZEN RULE block and never sets the status. Seen when decided: no RL update had run on either model; only
the NDIF pre-data scan (base rates, row norms) and the parent 0.6B runs existed. On the parent runs S7 gives per-seed
slopes -0.911 / -0.960 (Vera's rebuild: -0.91 / -0.96).

## D1: code fix in analyze_8b.py, pre-data (Felix, 06:05 PST; Kasper #22646-22647)

Two bugs in the frozen analysis code, found by Kasper before any RL update. The FROZEN RULE text is unchanged; the
code now does what that text says. (1) The main-arm filter mapped cue_alpha 0 to 1, so the C2 run (cue = 0) could
enter the fits; is_main() now requires arm 'cue', a size index and a non-zero cue. (2) An unswitched run that stopped
before B (cut at the wrapup or by a laptop self-stop) was scored as censored at B; it is now 'incomplete': censored
at its own last eval in the Tobit fit, and left out of the censored count and of the threshold clause, since the
frozen text censors at B only runs that reached B. Seen when decided: no RL data on either model. Tests on parent
runs plus synthetic C2/early-cut cases: C2 excluded, cut run counted as incomplete (n_cens 2, incomplete 1).

## D2: the job runs past a failed G2(b); run EXPLORATORY (Tove decides #22696, 06:13 PST)

RULE CHANGE: the job ran past a failed G2(b) (frozen: a failed gate stops the job). Status capped at UNCONFIRMED by
the frozen rule itself; run marked EXPLORATORY (front matter lab_exploratory: true).

What failed (data/g2_diag.json, Bruno #22679, Iris #22691): the bf16 tail on our streamed MPS cache vs NDIF's
Qwen3-8B gives max |d ln P(X)| 0.219 (dev0445), 3 of 64 dev prompts above 0.05, median 0.0013, top-1 63/64 (the miss,
dev0429, is an exact tie in NDIF). G2(a) passes (cache vs NDIF block-34 input rel err max 0.0045). The port itself
reproduces NDIF: the bf16 tail on NDIF's own input gives max 0.027, 0 of 64 above 0.05. The same NDIF input through
the tail in fp32 instead of bf16 moves ln P(X) by up to 0.140 (dev0410). So the frozen 0.05-on-every-prompt threshold
sits below the model's own precision floor; it is not a port bug (Tove #22696, Kasper #22693).
What did not change: the cache (no NDIF-cache swap, which would be re-selection after a failed gate), the frozen
analysis (analyze_8b.py), the status rule, every secondary read, the mini order rule and the laptop jobs.
Seen when decided: the gate numbers above and the step-0 scan (alphas for s_0..s_5, |gap(0)| 0.0192 ... 0.0006).
No RL update had run on either model. Consequence: the post can be UNCONFIRMED only; the oracle queues a replication
whose G2 threshold is calibrated before freezing (from the fp32-vs-bf16 floor on one input).
