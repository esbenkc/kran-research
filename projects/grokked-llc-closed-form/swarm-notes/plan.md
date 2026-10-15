# FINAL PLAN — theory-1015-grokked-llc-closed (plan, 05:58 PDT = 14:58 CPH)

Deadline 20:00 CPH = 11:00 PDT. PR open by 10:00 PDT. Merge (if CONFIRMED/NULL) by 10:40 PDT.

## Fixed names
- slug + figure id: `grokked-llc-closed-form`
- code: `kran-research/projects/grokked-llc-closed-form/`
- raw data: `kran-research/projects/grokked-llc-closed-form/data/`
- post: `blog.kran.ai/_notes/grokked-llc-closed-form.md`
- paper title to cite: "A Basin-Selection Perspective on Grokking via Singular Learning Theory" (Cullen, Estan-Ruiz, Danait, Li; arXiv 2603.01192 v3). v3 numbering: Thm 4.1 (secant), Cor 4.2 (under/over formulas), Thm 5.2 + Cor 5.3 (lazy memorising), Cor 5.4 + Remark 5.5 (Fourier/K_eff). The goal's "Thm 4.1/4.2/5.2/5.5" map to these. Use v3 numbers in the post, say so once.

## The ONE claim (pre-registered)
Prediction under test: Remark 5.5, grokked-basin LLC = K_eff (3p-1)/2.
Test: SGLD LLC (Lau et al. 2308.12108) at grokked checkpoints, p in {7,11,17,23,31}, K in {2p,4p}, 3 seeds.
- CONFIRMED if grokked λ̂/λ_paper lies inside the estimator's calibrated error band (band = max rel. error of the same SGLD settings on known-λ cases, floor ±15%) for ≥80% of (p,K) cells and the log-log slope of λ̂ vs λ_paper is in [0.85,1.15].
- NULL if grokked points sit outside the band with a consistent sign, or slope outside that range.
- Report bias = median λ̂/λ_paper per p, and whether its sign flips with p (goal asks this).
Memorising checkpoints are plotted too (second marker), prediction Thm 5.2 λ = p·min(l,K)/2, l = rank of centred train features at the checkpoint.

## Known theory issue (skeptic #617, plan confirms)
One-hot input ⇒ pre-activation u_a + s_b ⇒ extra symmetry u→u+c, s→s−c per neuron ⇒ generic rank K(3p−2), not K(3p−1); rank capped by p(N−1) (centring). Over-param value p·d(d+1)/4 is unreachable. Consequences:
1. Calibration does NOT use the paper's Cor 4.2 values as truth. Truth for calibration = exact rank(J)/2 at planted generic true solutions (smooth points), plus V-only linear models (λ = p·rank(F̃)/2 exactly). Report paper's Cor 4.2 value next to each, as a side result.
2. Dead neurons (w,v≈0 from weight decay) are singular points; they can add λ even if K_eff excludes them. Expect λ̂ > K_eff(3p−1)/2 possibly. Do not "fix" this; measure it.
3. At every checkpoint also compute deterministic rank(J)/2 of centred train outputs (a second, noise-free number) — goes in data, used in margin note, not a second figure.

## Conventions (everyone)
- L_n(w) = (1/N) Σ_i ½‖P⊥-centred residual_i‖² on TRAIN set only, NO L2 term (L2 = prior; localisation γ plays that role).
- SGLD: w ← w − (ε/2)(nβ ∇L_n + γ(w−w*)) + N(0,ε); λ̂ = nβ (E[L_n] − L_n(w*)); default β = 1/log N; nβ, ε, γ fixed once from calibration then frozen for all checkpoints. 4 chains, report mean and chain s.e.
- float64, CPU torch or numpy. Seeds 0,1,2.
- K_eff: neuron j active if ‖v_j‖·‖w_j‖² ≥ 1% of max over j; K_eff = #active. Also save Fourier purity (top-frequency power share of u_j, s_j) per neuron.
- Checkpoint file: `data/ckpt/p{p}_K{K}_s{seed}_{mem|grok}.npz` with W (2p×K), V (p×K), train_idx, test_idx, step, train/test acc + loss, lr, wd, frac.
- Results table: `data/llc.csv` cols: p,K,seed,phase,N,lambda_hat,lambda_se,K_eff,l_rank,rankJ,lambda_paper,lambda_shift(=K_eff(3p−2)/2),lambda_rank(=rankJ/2),eps,gamma,nbeta,chains,steps,train_loss,test_acc.
- Calibration: `data/calibration.csv` (case,p,K,truth,lambda_hat,rel_err,eps,gamma,nbeta) + `data/calibration.json` (frozen hyperparams).

## Tasks (claim before work)
| task | owner | output | ETA PDT |
|---|---|---|---|
| T1 training recipe + sweep + checkpoints + K_eff | build | data/ckpt/*, data/train_log.csv, src/model.py, src/train.py | recipe 06:40, sweep 07:30 |
| T2 SGLD estimator + calibration on known-λ cases (planted teacher rank/2, V-only linear), freeze hyperparams | plan | src/llc.py, src/calibrate.py, data/calibration.* | 07:15 |
| T3 run LLC + rankJ over all checkpoints → data/llc.csv | build (uses src/llc.py + calibration.json) | data/llc.csv | 08:15 |
| T4 figure (log-log λ̂ vs λ_paper, y=x, marker=phase, colour=p) + equation checks (sympy) | plan | src/figure.py, figures/, blog assets/figures/grokked-llc-closed-form.* , checks/equations.py | 08:40 |
| T5 research.md (paper defs: l, d, L2, their LLC hyperparams; prior LLC-vs-closed-form) | research | swarm-.../research.md | 06:20 |
| T6 post text + front matter | write | _notes/grokked-llc-closed-form.md | draft 08:50 |
| T7-9 independent derivation + rerun from data + null test + novelty → skeptics/<name>/ (signoff.md, novelty.md) | skeptic, skeptic2, skeptic3 | | 09:30 |
| T10 gates PREFLIGHT 0-3 + STYLE + preview render | critic | | 09:50 |
| T11 commit/push branch, PR, preview, merge | write (blog) + build (kran-research) | | 10:00–10:40 |
| T12 hit list append (Next posts) | research | /Users/esben/odin-context/oracle/hitlist-theory.md | 09:30 |

Null tests suggested (skeptics split, avoid duplicates): skeptic = SGLD on random-init (untrained) net; skeptic2 = regular quadratic with λ=d/2 + step-size sweep; skeptic3 = different SGLD chain seeds + nβ sweep on grokked checkpoints.
