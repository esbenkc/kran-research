# megaprobe-after

Question: **After a fine-tune hides false statements from a one-layer probe, does a probe that reads every layer still
catch most of them?**

No model is trained here. We take the three lambda = 0 LoRA adapters of `max-pool-survives` (room
theory-1019-max-pool-survives; LoRA r 8 on blocks 0-13 of Qwen2.5-1.5B-Instruct, trained to push a frozen mean-pooled
"false statement" probe at `hidden_states[14]` down), read every layer of the clean and the fine-tuned model on the
Geometry of Truth cities statements (Marks & Tegmark, arXiv 2310.06824), fit monitors on the CLEAN model only, and
score the fine-tuned model with a new 1% alarm line.

Frozen plan: `PREREG.md` (commit ad300da, 15:00:34 CEST). Later changes: `DEVIATIONS.md` (D1 implementation details
before any fit; D2-D3 exploratory controls and D4 figure layout, all after the first result).

## Reproduce (Mac mini M4, torch MPS, bf16; ~15 min; one heavy job at a time)

```bash
cd kran-research/projects/megaprobe-after
./run_extract.sh                         # extract.py --state clean|seed0|seed1|seed2 (~55 s each, peak footprint 6.3 GB)
uv run python gate.py                    # reproduction gate vs the 1019 logs -> data/gate.json
uv run python probes.py pool             # mean-pooled features -> data/pooled/ (2 GB, not committed)
uv run python probes.py select           # PREREG s4 grids on clean -> data/hp.json (~5 min)
uv run python probes.py curves           # label-count + convergence fits on clean -> data/fits/
uv run python probes.py score            # final fits, every state scored -> data/scores.npz (~2 min)
uv run python analysis.py                # every pre-registered number -> data/results/results.json
uv run python controls.py                # D2 exploratory controls -> data/results/controls.json
uv run python d3.py                      # D3 country-token probe at every layer + megaprobe fit seeds -> data/results/d3.json
uv run python check_equations.py         # numeric checks of the post's two displayed equations
uv run python ../../projects/megaprobe-after/run.py   # (from kran-research/) the figure
```

`data/acts/` (9.6 GB of per-token activations, every layer, fp16) is not committed; `data/acts_sha256.txt` holds the
checksums. fp16 storage is lossless here: the model runs in bf16, and every bf16 value in range is exact in fp16.

## Setup

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template |
| fine-tuned states | `data/ref1019/runs/lam0__seed{0,1,2}__adapter` (copied from max-pool-survives) |
| readout | `hidden_states[l]`, l = 0..28 (0 = embedding, l >= 1 = output of block l-1, before the final norm), tokens 1..n-1 |
| labels | y = 1 for a FALSE statement |
| splits (by city, the 1019 split) | fit pool = probe + attack + stop (1400 statements, 350 cities); line = 200 refit TRUE statements; eval = 1192 (298 cities) |
| alarm line | 2nd-highest score of the 200 line statements of the scored state; flag iff score > line |
| CI | city-cluster bootstrap over the 298 eval cities, 2000 draws, seed 1020, lines fixed |

## Monitors (all fit on clean activations only)

| id | readout | model, hyperparameters (data/hp.json) |
|---|---|---|
| `base` | L14 mean over tokens | the frozen 1019 probe (the fine-tune's target) |
| `lin<l>` | layer l mean over tokens | StandardScaler + LogisticRegression(L2), C by 5-fold GroupKFold-by-city OOF AUROC from {0.01, 0.1, 1, 10} (ties within 0.002 -> smaller C); chosen C = 0.01 (l 2-28), 0.1 (l 1) |
| `mlp14` | L14 mean over tokens, standardised | 1536-256-1 ReLU net, Adam lr 1e-3, batch 64, weight decay from {1e-4, 1e-3, 1e-2} by 4-fold OOF AUROC (ties -> larger), early stop on a 20% city-grouped inner split (BCE, patience 20); chosen 1e-2 |
| `mega` | every token at l = 1..28 | per layer, whitened top-k PCA of the clean fit-pool tokens; one softmax over all (token, layer) pairs of q_l.z + c_l weights the score w_l.z; Adam lr 1e-2, L2 on q and w, 400-row minibatches, 300 epochs, checkpoint every 10 epochs by inner AUROC (20% city-grouped inner split); k x L2 from {32, 128} x {1e-3, 1e-2, 3e-2} by 4-fold OOF AUROC (ties -> smaller k, larger L2); chosen k = 32, L2 = 1e-2 |

The megaprobe is the `megaprobe-spin-lies` architecture (`fit_c`, kran-research branch
swarm/oracle-1020-megaprobe-spin-lies), applied to statements instead of answers.

## Files

| path | what |
|---|---|
| `common.py` | config, model load, tokenisation, split/probe loaders, threshold rule, `Acts` memmap reader |
| `extract.py`, `run_extract.sh`, `watchdog.sh` | per-token activations at every layer + per-token log-probs; PID-bound footprint watchdog |
| `gate.py` | reproduction gate (frozen probe vs 1019 logs) |
| `probes.py` | pooled features, selection grids, label-count curves, final fits, scoring |
| `analysis.py` | catch rates, CIs, twin-paired separation per layer, retention, knowledge check, attention mass, status, prereg outcome |
| `controls.py` | D2 exploratory controls (one-layer megaprobe at l18/l22; all-layer concatenated linear; eval-true line sensitivity) |
| `d3.py` | D3 exploratory: linear probe on the last country-name token at every layer; megaprobe fit seeds 1021/1022 |
| `check_equations.py` | the post's d' and megaprobe formulas recomputed in numpy -> data/results/check_equations.json |
| `run.py`, `plot.js` | the figure (two panels after DEVIATIONS D4: monitor strip + per-layer catch for mean-pooled and country-token probes) |
| `data/scores.npz` | every monitor's score of every statement in every state; 5 shuffled-label nulls (lin14, mega); label-count fits; mega attention per layer and per token |
| `data/hp.json`, `data/fits/` | chosen hyperparameters + grids; fitted parameters (clean PCA, final mega, mlp14, curve fits) |
| `data/results/results.json` | every pre-registered number |
| `data/logs/` | stage logs + footprint logs |
