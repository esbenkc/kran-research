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

## Queue order (order only, no rule change; Felix #22703, Tove #22704, 06:14 PST)

Measured mini rate 0.64 s/update (dense evals on), peak 7.23 GB alone (Bruno #22702). The frozen order rule (#22604)
gave branch 2 (3 seeds + 2 controls ~4.5 h > 4.0 h to the 10:15 PST last start). Since D2 caps the status at
UNCONFIRMED, Tove re-ranked by information per hour before any 8B result was read: (1) 8B seed 0 at all 6 sizes,
(2) paired NDIF-cache cell k=2 seed 0 (data/runs_explore), (3) matched 0.6B arm seeds 0-2, (4) 8B seeds 1-2,
(5) no-offset cells s_0, s_3 seed 0 (data/runs_explore), (6) C1 seed 0 at the full 8,192. C1 seed 1, 8B seeds 3-4 and
0.6B seeds 3-4 run on the laptop queue (labq) for the replication. Last start 10:15 PST; runs still going at 10:45
are stopped and scored as incomplete.

## D3: C1 seed 0 killed on a false stall read and resumed (procedure only, no rule change; Bruno #22833-22834, logged by Felix 09:25 PST)

C1 seed 0 (mini, shuffled coin, full 8,192) was killed by Bruno at 09:22 PST on a false stall read: its log file had
been replaced in the shared clone by a git checkout/autostash while the process kept writing to the unlinked old
file. The run was healthy (checkpoint at update 4,096 at 09:20, ~0.59 s/update). It was resumed from the update-4,096
checkpoint at 09:24 PST with the same arguments. Resume restores LoRA, AdamW state, numpy and MPS generator states but
is not bitwise (0.6B test: identical for ~230 updates, then one rare sampled token differed); the run JSON carries
resumed_from [4096]. Seen when decided: C1 seed 0's curve up to update 4,096 (no switch), all 8B and 0.6B main cells.
Lesson applied: never git-add a running job's log.

## D4: code fix in analyze_8b.py: only finished seeds get a slope (Felix, 10:01 PST; no rule change)

The frozen text says 'the status uses every finished seed'. analyze_8b.py gave a slope to any seed with >= 4
uncensored sizes, so the matched 0.6B seed 3, still running at 10:00 PST (sizes k0-k3 only), entered S1 with a slope
of -0.725 fitted on the four largest cues alone, where both models' curves are flattest. per_seed() now gives a slope
only to a seed with a run at every planned size and no incomplete run (finished=True in the output). Effect at 10:00:
matched 0.6B back to seeds 0-2 (-0.917 [-1.025, -0.809]); S1 +0.013 [-0.076, +0.101]. The 8B arm (3 finished seeds)
and its status are unchanged. Seen when decided: all 8B cells, C1 seed 0 (no switch), 0.6B seeds 0-2 and seed 3 k0-k3.

## D2 correction (Felix, 10:08 PST; critic2 #22883-22885, Vera #22887)

D2 quoted only the bf16-tail reading of G2(b). The gate as run (data/gates.json, tail in fp32 as trained) missed on
6 of 64 dev prompts (max |d ln P(X)| 0.129). The bf16 tail on our cache missed on 3 of 64 (max 0.219), and the fp32
tail on NDIF's own input missed on 2 of 64 (max 0.140) (data/g2_diag.json). The decision and its consequence in D2 do
not change.
