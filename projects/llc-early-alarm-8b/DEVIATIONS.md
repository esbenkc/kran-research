# DEVIATIONS — llc-early-alarm-8b

Frozen rule: `PREREG.md` (commit 1b81a5f3e, pushed 2026-11-24T14:07:02Z; header fc37c0966). Every line below was
written before the run it describes. None changes the status rule, a threshold or the primary metric, so none is a
"RULE CHANGE:" line.

## D1. The gate failed (frozen outcome, not a deviation; recorded here for the timeline)
Bruno posted the (c, gamma) grid and the selection rule at 2026-11-24T14:08:18Z (room #29971), before the grid ran.
Result (commit 7b9f86369, `data/gate/grid.json`, `frozen.json`, `select.log`; room #29980): 0 of 20 settings pass.
Best: c 0.3, gamma 10: 196.9 (-16.6%) at rank 1 and 276.5 (-11.4%) at rank 8 (Aoyagi 236 / 312). Per FROZEN RULE (9)
and (6): stop; status UNCONFIRMED; no scored 8B SGLD estimate runs. The prediction (which included "the gate
passes") fails.

## D2. Exploratory add-ons after the gate failure (unscored; Felix #29987)
None of these feeds the status. Each is labelled EXPLORATORY in the proof matrix and in the post.
- E1 (budget sweep, CPU, `gate_budget.py`, started by Bruno at about 06:10 PST, a minute before this file was pushed;
  its docstring labels it exploratory): the same frozen gate case and `sgld.py` with longer chains. Steps per chain S in
  100 / 400 / 1600 / 6400 (burn S/4, draws 3S/4), 2 chains, c in {0.03, 0.1}, gamma in {0.1, 1}, chain-seed repeats
  0-2 -> `data/gate/budget.json`, `budget.log`. Reports the smallest budget at which the estimate is within 10% of
  Aoyagi at both ranks. Vera's independent rebuild (#29986, `skeptic/vera_gate.py`) saw 2 x (1000 burn + 3000 draws)
  at c 0.1, gamma 1 land within 0.1-3%. (Felix #29987 also proposed n*beta 26.4 vs 1000 at the longest budget; that
  cell runs only if Bruno adds it, and gets its own line here first.)
- E2 (8B transfer test, MPS, after J1): the frozen 8B estimator on 2 real checkpoints (k2 seed 5 at updates 32 and 88)
  at c x {1/3, 1, 3} (6 estimates): does the 8B estimate scale with c as the gate's does?
- E3 (no SGLD): the race readouts D(t), ||BA||_F and the training gradient norm at the frozen grid points, scored
  against the 3 C1 seeds with the frozen rule (4)/(5) by `analysis.py`. Reported in one sentence and the proof matrix,
  not in the title.
