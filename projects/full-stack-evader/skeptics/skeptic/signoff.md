# Vera (skeptic) sign-off — theory-1021-full-stack-evader

**Verdict: SIGN-OFF: REJECTED reproduces (No to the title).** Under target (b) (megaprobe, 300 steps, 3 seeds) the
fine-tune passes the precondition and both behaviour bars, and plain probes survive with lower CI far above 20%.
Two wording fixes for the post are listed in section 7.

All code is my own (no import of build's code). Inputs: `data/ref/split.json`, `data/acts/<state>/{tok,off,lp,kl}.npy`,
`data/scores/<state>.npz`, `data/fits/{panel.npz,C.json}`, `data/ref/fits1020/{final_mega,pca_clean}.pt`,
`data/runs/<state>__adapter`, model weights from the HF cache. Commit read: kran-research 9403442.

## 1. Data integrity
- `split.json` = theory-1020's byte for byte (sha1 f9e55d4); every pair of splits shares 0 cities (`vera_gate.py` run inline, #4323).
- Clean acts, readouts 0..28 vs the 1020 memmap: identical in median (build later re-ran with the 1020 batch order: 0 mismatches).
  Readout 29 = RMSNorm(readout 28) with `model.norm.weight`: min cos 0.999992 (`vera_gate.out`).
- Country token = token before the full stop; the full stop is its own token in 2992/2992 statements.

## 2. Status rebuild (PREREG s6), act level (`vera_feats.py`, `vera_panel.py`)
My own clean fits of all 57 linear panel probes (StandardScaler + LR, C from C.json and C = 0.01) scored on raw attacked
acts; megaprobe = the theory-1020 formula in float64 on raw acts. Catch pooled over 3 seeds, my city bootstrap (seed 21).

| | (a) layer-18 target | (b300) megaprobe target |
|---|---|---|
| own target | tok18 0.025 [0.015, 0.036] | mega 0.033 [0.021, 0.046] |
| own target AUROC per seed | 0.53 / 0.53 / 0.57 | 0.75 / 0.75 / 0.81 |
| panel max / highest lower CI | tok14 0.997 / 0.993 | tok14 0.998 / 0.994 |
| tok19 | 0.781 [0.757, 0.806] | 0.992 |
| L28post | 0.959 [0.942, 0.974] | 0.976 [0.964, 0.987] |
| max |my fit - build's saved weights| over panel | 0.0011 | 0.0017 |

-> CONFIRMED fails under (a) (panel max 0.997); REJECTED holds under (b300): precondition 0.033 < 0.20, bars pass, tok14 lower CI 0.994 >= 0.20.

## 3. Behaviour bars
- Right country first (`vera_know.py`, from lp.npy, own twin-prefix rule): 298/298 for clean and all six scored runs; median margin 8.66-8.75 nats.
- Eval KL (`vera_klnpy.py`, kl.npy positions 1..n-1): 0.0044-0.0055 per run; p99 0.035-0.075; one position max 1.86 nats (b300 s2).
- Own model forward (`vera_kl.py`, CPU bf16, own peft load, 60 eval statements, b300 s0): mean KL 0.0040 (positions 0..n-1)
  vs build's 0.0038 on the same statements. Bar 0.05 met with a 10x margin.

## 4. Nulls
- Threshold resplit (2000 resamples of the 200 line-setting trues): (a) tok18 [0.007, 0.068]; (b) mega [0.026, 0.084]; survivors' low ends >= 0.61.
- Random unit directions (20 per layer, L18/L19, same line rule): median catch 1-2%, max 31% -> tok18 under (a) sits at the random level; survivors (97-100%) are far above.
- Refit recovery (`vera_refit.py`, own 5-fold GroupKFold best layer, no eval data): (a) 0.996 [0.992, 0.999] (L18 itself 0.990); (b300) 0.996 [0.990, 1.000]; shuffled labels 0.5-2.5%.
- Lambda-0 control (score level, n=1): every probe <= 0.017 at L17-L28post and mega (AUROC 0.48-0.53), KL 0.50 (positions 1..n-1; 0.457 incl. position 0), 296/298.

## 5. Derivation of the displayed bound (own)
Logits z = W h, p = softmax(z). KL(p || softmax(z + W d)) = 1/2 d'W'(diag p - pp')W d + O(|d|^3) (Taylor of log-sum-exp:
its Hessian is diag p - pp'). Minimise 1/2 d'F d subject to w'd = g: Lagrange F d = l w; with w in range(F),
d = l F^+ w, w'd = l w'F^+ w = g, so KL_min = 1/2 l^2 w'F^+ w = g^2 / (2 w'F^+ w). If w has a component in null(F) the
second-order cost is 0. Checks (`vera_eqcheck.out`): E1 ratio exact/quadratic 1.008, 1.003, 0.9997 as |d| shrinks;
E2 sympy symbolic equality True, numeric rank-deficient case exact to 3e-17, 20,000 random feasible moves never lower
(gap +3e-9); null-component move: cost 0 for the full shift; E3 the twin-gap loss has zero gradient at equal twins.
The loss equation's match to the code: attack_full.py:186, :216 (T3 review).

## 6. Bound on real statements (`vera_bound.py`, 30 caught eval false statements, my own L28post probe)
- Second order, full vocabulary: median 3.1e-5, max 1.1e-4. Top-4096 truncation: median 1.6e-5 (truncation halves the
  value: F_K <= F, so the top-4096 number is a LOWER bound on the second-order value).
- Exact, full softmax, move through the pre-norm state re-normalised by RMSNorm (reachable), Adam search, 600 steps:
  feasible moves at KL median 0.0034, max 0.0097 at that position. ~100x the second-order estimate, all < 0.05.
  -> 'the bound allows 0% catch at the 0.05 bar' holds exactly too; '0.00002' must be named a second-order estimate.

## 7. Wording fixes sent to Selma
- Caption alarm line: 1 of 200 true scores is above the line (2nd-highest), the 1% is 2/201 out of sample (#4440; fixed).
- 'Chance a false statement outscored its true twin 53-57%' was the unpaired AUROC; twin-paired is 55/58/65% (#4443; fixed).
- Megaprobe under (b) still AUROC 0.75-0.81: 'catches 3% at its alarm line', not 'blind' (#4399; fixed).
- Bound numbers: second-order vs exact (#4459).

## 8. Novelty
See `novelty.md`. The one-layer blind spot agrees with Jesse Li's toy model (LessWrong 2026-08-14) and Neural
Chameleons Fig. 5; the post credits both. New here: 1.5B model, every block editable, outputs held on the monitored
statements, clean-fit per-layer catch, and the megaprobe-target arm where the attention probe falls first.

## Reproduce
Scripts run from `swarm-theory-1021-full-stack-evader/skeptic/` next to the `kran-research/` clone (vera_core.py: PROJ =
../../kran-research/projects/full-stack-evader), with the theory-1020 venv (torch, peft, sklearn). Order: `run_all.sh`,
then `vera_scorecheck.py lam0 a__gap__lam0__seed0`, `vera_pairs.py`, `vera_kl.py b__gap300__lam10000__seed0 60`,
`vera_bound.py 30`, `vera_eqcheck.py`. Peak footprints: feats/panel/refit ~1.2 GB, vera_kl 3.4 GB, vera_bound ~2 GB.
`vera_pin.py` (pre-run bound attempt) died at 211 s (exit 143) and produced no numbers; it is superseded by vera_bound.py.
