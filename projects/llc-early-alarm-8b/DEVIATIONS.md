# DEVIATIONS — llc-early-alarm-8b

Frozen rule: `PREREG.md` (commit 1b81a5f3e, pushed 2026-11-24T14:07:02Z; header fc37c0966). Each exploratory run's
line was pushed before that run started, except E1 (see its line); result lines come after. None changes the status rule, a threshold or the primary metric, so none of
them is a rule-change line.

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
- E4 result (Bruno #30020, data/gate/e4_psgld.json): diverged at all 10 settings (estimates 1e13 to 1e17), 0 of 10
  pass. Bruno's reading: at w* the unused hidden units have exactly zero loss gradient, so with a = 1e-8 the
  preconditioner G reaches about 1e8 and the injected noise is large in exactly the singular directions.
- E4b NOT RUN (Tove #30023, Felix #30026): a retune of the stabiliser after seeing E4 diverge would be a fishing trip.
  A tuned preconditioned sampler, frozen on the gate before any 8B run, goes to the hit list for the follow-up.
- E2/E2b precision and memory (Bruno #30105, logged in every data/llc/e2/*.json as head_dtype): the walks ran the 2
  tail blocks in fp32 but the output head in bf16 (llc.py --head-dtype bf16) to fit the 5 GB share; RL training used an
  fp32 head. The checkpoint's reference distribution and the walk use the same bf16 head, so L(w*) = 0 still holds
  (||grad L(w*)|| 4.4e-7 to 4.6e-7). The six E2 estimate processes each peaked at 5.53 GB footprint for about 2.5 min,
  0.5 GB over the cap (data/mini/E2/chain.log), disclosed in the post's methods.
- E2b memory change (Vera #30068, before E2b runs): E2's estimate processes peaked at 5.53-5.54 GB footprint (over the
  5 GB cap; RSS ~0.35 GB). For E2b only, the SGLD noise is drawn on the MPS device (a torch generator on mps) instead
  of on the CPU and copied, so E2b's noise stream differs from E2's. Its first 25 steps run under /usr/bin/time; if the
  peak is still above 5.0 GB, E2b drops to 2 x 1,600 steps (burn 400) instead of 2 x 3,200. E2's saved numbers are
  unchanged.
- E2b as run (Bruno #30075): the 25-step memory test peaked at 4.46 GB (data/mini/E2b/memtest.log), so E2b runs the
  full 2 x 3,200 steps (burn 800), started 06:30:39 PST, noise from the device (MPS) generator seeded 1000 x step + k.
  The launcher logged commit d0b19ff34 but ran the working tree that was committed one minute later as 683e931e3
  (llc.py --noise-on-device); the code that ran is 683e931e3.
- E2 result line (Bruno #30075, data/llc/e2_summary.json): h_max at update 88 = 121.82 vs 47.92 at update 32 (x2.54,
  both converged). Under the frozen 'eps fixed per size from update 32' rule, update 88 therefore runs at an effective
  step constant 2.5 times larger than update 32, so any u88-minus-u32 difference mixes the change in the model with a
  change in the effective step. Readings u32 / u88: 1.31 / 2.07, 26.26 / 38.73, 22.16 / 35.76 at c 0.033 / 0.1 / 0.3.
- E2b PREDICTION (Felix #30047, posted 2026-11-24T14:23Z, before E2b ran; exploratory, not a frozen rule): per Hessian
  direction i the localised SGLD variance relaxes like (1 - eps a_i)^t, a_i = n beta h_i + gamma, so directions at half
  weight (n beta h_i = gamma) need about S* = (n beta h_max + gamma) / (2 c gamma) steps per chain. Gate check (h_max
  3.55, c 0.1, gamma 1): S* = 473, vs E1's first pass at 400-1,000. 8B (h_max 47.92, c 0.1, gamma 1): S* = 6,326, so
  E2b (2 x 3,200 steps, burn 800) is predicted NOT to plateau. Refuted if E2b's running estimate is flat within its
  batch-means SE from about 1,600 steps on.
- E2b result (Bruno #30137, data/llc/e2b/m8_k2_s5_cue-m8_k2_s5_u00088_c0.1_long.json, commit 3554a0a32; 1,916 s,
  peak 4.45 GB): LLC-hat 12.08 (chains 12.05 / 12.12, batch-means SE 0.53). Running estimate at 100 / 500 / 900 /
  1,600 / 3,200 steps: 35.5 / 14.3 / 12.6 / 12.0 / 12.1; draws 801-2,000 = 12.01 (SE 0.43) vs 2,001-3,200 = 12.15
  (SE 0.96). By its own stated criterion ('flat within its batch-means SE from about 1,600 steps on') the E2b
  PREDICTION above is REFUTED: the estimate fell (not rose) from the 100-step value and was flat from about 900 steps,
  well before the predicted S* = 6,326. A flat running mean does not prove the slowest directions have settled.
- Correction to E1's summary (Vera #30000): at c 0.1, gamma 1 the gate case is already within 10% at 400 steps per
  chain in Bruno's data (r1 -6.5%, r8 -7.8%) and passes from 1,000 in Vera's own data; at 1,600-6,400 steps it sits
  2-5% high.
- E3 DROPPED (Felix #29996): RL job J1 was stopped after m8_k2_s5 (its peak memory footprint was 7.85 GB, above the
  5 GB cap while another swarm is open; Vera #29994), so the race readouts at the other runs do not exist. The race
  was a wording rule for an LLC alarm that never ran (Kasper #29991). m8_k2_s5 matched the parent run bit for bit
  (Bruno, commit d160468a8). A J1 relaunch at 06:13:17 PST began m8_k2_s6; Bruno killed it at 06:14 after update 24
  (#30003); its partial adapters in raw/lora/m8_k2_s6/ are not used. No C1 run exists.
