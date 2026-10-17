# PLAN — algzoo-432 (owner: Felix/plan)

Model: `example_2nd_argmax()` = M_16,10, 1-layer ReLU RNN, no bias, 432 params, acc 95.3% vs truth.
Goal (Esben): hand-written algorithm that matches the MODEL's argmax on >=99% of N(0,1)^10 inputs,
every parameter block explained.
Not our claim: ARC's open challenge (mechanistic accuracy estimate that beats sampling in MSE). Final
summary must say this plainly.

## Frozen metric (PREREG.md, Rasmus writes and owns)
- Agreement = mean[argmax alg(x) == argmax model(x)], x ~ iid N(0,1)^10, N=1e6, seed 432432 (held out).
- All constants in any algorithm are fit or tuned on other seeds only (dev seed 0..99). Report 95% CI.
- Also report agreement on the model's error set (where model != true 2nd argmax), n >= 20k.
- No tautology: `algorithm.py` must not import the model or run the raw 16x16 recurrence.

## Structure found so far (Felix, rms-scaled weights, 200k samples)
- Subcircuit A = neurons {1,2,4,6,7,8}: rows read only from columns {1,2,4,6,7,8} (others < 0.05 rms-scaled,
  row 4 has small 0.1-0.3 leaks). Contains ARC's running max (2), max-increment (4), LOO-max minus last
  input (6,7,1); neuron 8 is new (reads 1,2,7,8), likely another LOO-max feature.
- Dense B = {0,3,5,9,10,11,12,13,14,15}: |W_hi|/rms ~ 0 except 9 (-0.26) and 14 (-0.10). They read A
  and each other. 12,13,14 active >= 98% (almost linear); 4 and 9 active ~11% (rare gates).
- Readout: largest rms-scaled W_oh columns: 1,6,7,8 (A), then 10,11,12,14,0,2.

## Two deliverables (DONE = either reaches >=99% with every block named)
- D1 feature algorithm: closed-form features of x (maxes over subsets, x_t, linear filters) -> linear
  logits -> argmax.
- D2 simplified RNN: rounded/sparsified weights, each kept entry named with a role, dropped mass shown
  irrelevant (ablation delta < 0.5 pt). Fallback if D1 stalls under 99%.

## Tasks (claim before work; reviewers in brackets)
| # | Task | Owner | Reviewers |
|---|------|-------|-----------|
| T1 | env, baseline 95.3%, weights_std.json, core.py (forward with hidden states, sampler, agree()) | Bruno | Kasper |
| T2 | research.md: Zihao Chen notes + colab, ARC neuron roles, handcrafted_2nd_argmax(10) construction | Iris | Kasper |
| T3 | PREREG.md + independent check_agreement.py (held-out seed, CI, error-set split) | Rasmus | Kasper |
| T4 | subcircuit A {1,2,4,6,7,8}: exact closed-form feature per neuron, R^2 per neuron per t, leaks quantified | Bruno | Rasmus, Kasper |
| T5 | dense B part 1 {12,13,14,10,11,15} (near-linear): express as linear filters over A-features/x | Felix | Rasmus, Kasper |
| T6 | dense B part 2 {0,3,5,9} (gated, incl. 9 reading 7 at -110): gate condition + role | Iris | Rasmus, Kasper |
| T7 | ablation ladder: per block (neuron, W_hh entry group, W_oh column) zero/mean/round -> agreement delta; pruning to D2 | Kasper | Rasmus |
| T8 | readout + integration: algorithm.py (D1 and/or D2), agreement via check_agreement.py | Felix | Rasmus, Kasper |
| T9 | EXPLANATION.md: per-block table (W_hi 16, W_hh 256, W_oh 160 -> role), tiers, limits | Iris | Kasper, Rasmus |

## Update 10:25 (Felix) — feasibility result, path = Option 3 (pending Esben, ask #1514)
Facts: 0.1% weight noise flips 1.2-2% of outputs; rounding all W to 3 sig digits -> 98.3%; W_hh rows of A
need exact gains (Rasmus: B rounded to 3 digits + A exact -> 99.2%). Oracle feature families cap ~95.5%;
even exact per-sample local Jacobians restricted to top-5 input values cap at 97.5%. Order-only ReLU-mask
state machines cap ~94%. So no compact algorithm reaches 99%.
Deliverables now:
- C1 `algorithm_compact.py`: mechanism algorithm (<=200 literals), measured agreement + error-class table.
- C2 `explained_model.py`: exact rewrite in interpretable coordinates (A feature basis, gate 9, B oscillator
  modes, readout as phase decoder); agreement 100%; every block labeled. Exceeds literal cap by design; the
  explanation value is the structure, reported as such.
- C3 block table (all 432 entries: role + ablation/noise sensitivity).
Task changes: T7 (ablation table, C3) -> Iris after T6. T10 (C2 exact rewrite) -> Bruno after T4.
T8 (C1) stays Felix. T9 EXPLANATION.md -> Iris last.

Order: T1, T2, T3 run now. T4-T7 start when T1 is done (core.py). T8 after T4-T6. T9 last.
Compute caps: max 2 worker processes per agent, total RSS < 5 GB. CPU numpy is enough (1e6 x 10 x 16).
Shared code only in core.py (Bruno owns it; ask before editing). Each task writes its own file:
`subA.py/subA.md`, `denseB1.*`, `denseB2.*`, `ablate.*`, `algorithm.py`.
Escalate to Esben (ask-esben) if D1 and D2 both stall under 99% after two different attempts.
