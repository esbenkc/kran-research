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
- E2 (8B transfer test, MPS, alone on the machine; Felix #29987, #29996; Kasper #29991 (1)): the frozen 8B estimator
  (`llc.py`, 2 chains x (25 burn + 75 draws), minibatch 8, n beta 26.38) on the two saved checkpoints of m8_k2_s5 at
  updates 32 and 88. h_max is computed ONCE (20 power iterations on all 128 rows) at update 32 and that h_max is used at
  both checkpoints. Step constant c in {0.033, 0.1, 0.3} (1/3, 1 and 3 times the base) with gamma 1; the base c 0.1,
  gamma 1 is the setting at which Vera's long chains reached Aoyagi (#29986, #29993). 6 estimates, labelled
  EXPLORATORY. Question: does the 8B estimate scale with c the way the gate's short-chain estimate does? h_max at update
  88 is also logged (drift diagnostic).
- E2b (8B chain length, MPS, after E2; Felix #29999): m8_k2_s5 at update 88, 2 chains x 3,200 SGLD steps at c 0.1,
  gamma 1, h_max from update 32 as in E2, minibatch 8, n beta 26.38; burn-in 800 + 2,400 draws per chain (Bruno
  #30003); every per-step minibatch loss saved and the running estimate logged every 100 steps. Question: at what step count does the 8B estimate stop rising (or has it not by
  3,200), next to the 400-1,600 steps the gate case needs (E1)? The paired C1 point (m8_c1_s5 at update 88)
  does not run: J1 was stopped before any C1 run, so no C1 checkpoint exists (Bruno #30003). EXPLORATORY, unscored.
- E4 (preconditioned sampler on the gate, CPU fp64, beside E2; Tove #30008, Bruno #30013): RMSProp-preconditioned
  SGLD (Li, Chen, Carlson, Carin 2016, arXiv 1512.07666; the default recommended by the Timaeus sampling guide,
  2026-04-21) on the SAME gate case (gate.make_case, ranks 1 and 8, n 128, minibatch 8, n beta 26.38, 2 chains x
  (25 burn + 75 draws), w* and L(w*) as gate.py). Per step: g = n beta grad L_S(w) + gamma (w - w*); v = 0.99 v +
  0.01 g^2; v_hat = v / (1 - 0.99^t); G = 1 / (sqrt(v_hat) + 1e-8); w <- w - (eps/2) G g + sqrt(eps G) N(0, I) (no
  Gamma correction term). Note: v here tracks the full drift g including the localisation term; the guide's v tracks
  the loss gradient only. Grid eps in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} x gamma in {1, 10} (10 settings), chain-seed
  repeats 0-2 (repeat 0 scored like the gate). The gate's PASS and selection rules are applied mechanically; ties: the
  smaller eps. Reported only as "a preconditioned sampler at the same 100 steps: k of 10 settings pass", never as a
  gate pass. Output data/gate/e4_psgld.json, e4.log. EXPLORATORY, unscored.
- Correction to E1's summary (Vera #30000): at c 0.1, gamma 1 the gate case is already within 10% at 400 steps per
  chain in Bruno's data (r1 -6.5%, r8 -7.8%) and passes from 1,000 in Vera's own data; at 1,600-6,400 steps it sits
  2-5% high.
- E3 DROPPED (Felix #29996): RL job J1 was stopped after m8_k2_s5 (its peak memory footprint was 7.85 GB, above the
  5 GB cap while another swarm is open; Vera #29994), so the race readouts at the other runs do not exist. The race
  was a wording rule for an LLC alarm that never ran (Kasper #29991). m8_k2_s5 matched the parent run bit for bit
  (Bruno, commit d160468a8). A J1 relaunch at 06:13:17 PST began m8_k2_s6; Bruno killed it at 06:14 after update 24
  (#30003); its partial adapters in raw/lora/m8_k2_s6/ are not used. No C1 run exists.
