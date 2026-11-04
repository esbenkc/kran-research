# transcoder-neuron

Does a GPT-2 transcoder match the real MLP at the logit only for features that copy a neuron?
Room `theory-1104-transcoder-neuron`. Frozen rules: `PREREG.md` (v2.6, commit 8ece4089). No model-API calls ($0).
GPT-2 small (`openai-community/gpt2`) + transcoders `pchlenski/gpt2-transcoders` (Dunefsky, Chlenski & Nanda,
arXiv 2406.11944), layers 8-11. Torch 2.15 MPS, fp32, Mac mini M4. Environment: `env/pyproject.toml` + `env/uv.lock`
(`cd env && uv sync`).

## Result (data/status.json)

Status **UNCONFIRMED** by the frozen rule: every layer is NEITHER (neither the YES nor a NO clause fires).
Prediction A failed, Prediction B failed (pooled partial Spearman -0.094 > -0.1), Prediction C failed (4/12).

| layer | DEFINED | NEURON R^2 [95% CI] | REST R^2 [95% CI] | GAP | real neuron | random push | verdict |
|---|---|---|---|---|---|---|---|
| 8 | 0.818 | 0.659 [0.62, 0.74] | -1.13 [-2.74, 0.08] | 0.658 | 0.174 | 0.25 / 0.19 | NEITHER |
| 9 | 0.843 | 0.943 [0.80, 0.96] | 0.591 [0.26, 0.77] | 0.855 | 0.787 | 0.32 / 0.27 | NEITHER |
| 10 | 0.787 | 0.447 [0.19, 0.61] | -0.148 [-0.56, 0.16] | 0.152 | 0.260 | 0.12 / 0.15 | NEITHER |
| 11 | 0.882 | 0.631 [0.36, 0.76] | -0.552 [-1.58, -0.03] | 0.644 | 0.408 | 0.25 / 0.49 | NEITHER |

(random push column: NEURON / REST cells.)

## Rebuild

```
cd projects/transcoder-neuron
env/.venv/bin/python gate.py                       # G1-G6 -> data/gate.json
env/.venv/bin/python run.py --layer 8 --smoke      # 40 units, ~1 min
./run_all.sh                                       # L8-L11, ~35 min, footprint 2.8 GB (watchdog cap 3.1)
env/.venv/bin/python gate_graph.py data/pushes/L8_smoke_pushes.csv   # G7 -> data/gate_graph_L8.json
env/.venv/bin/python analyze.py                    # -> data/status.json
env/.venv/bin/python proof_matrix.py               # -> proof-matrix.md (keeps skeptic rows)
env/.venv/bin/python oracle_copy.py                # -> data/oracle_copy.json
env/.venv/bin/python zstar.py                      # -> data/zstar.json
```

run.py caches the unit/context selection in `data/pushes/L{L}_selection.pt`; delete it to redraw from scratch
(same seeds give the same draw).

## Files

| file | what |
|---|---|
| `common.py` | hand-written GPT-2 forward from the HF weights with MLP-input / MLP-output pushes, the circuit-tracer linearisation (`linearise=L`: attention patterns + LN scales detached, downstream MLP = frozen error + transcoder Jacobian, `_TCLin`), the transcoder loader, neuron cosines |
| `data.py` | held-out text loader (gate only) |
| `gate.py` | G1 own forward vs HF logits; G2 linearised forward value == real; G3 transcoder FVU + spliced loss; G4 zero/constant pushes; G5 `_TCLin` grad vs explicit; G6 real grad vs finite difference |
| `gate_graph.py` | G7 graph gradient vs finite differences through an explicitly frozen replacement model |
| `run.py` | one layer: classes, sampling (PREREG Sampling bullet), READ and WRITE arms, all rows; column glossary in its `COLUMNS` string |
| `analyze.py` | PREREG v2.6 status, Predictions A/B/C, named rows, class-label shuffle, tail shares |
| `proof_matrix.py` | `proof-matrix.md` |
| `oracle_copy.py`, `zstar.py` | diagnostics for the real-neuron row (Tove #14439, Felix #14440) |
| `data/pushes/` | raw: `L{L}_units.csv` (one row per unit), `L{L}_pushes.csv` (one row per unit x context), `L{L}_meta.json` |
| `data/windows.npy` | the 2,000 held-out windows (pile-10k docs 5000+, BOS + 127 tokens), rule in `windows_meta.json` |

## Definitions (see PREREG.md)

- Push: `du = -pre * e / |e|^2` added to `blocks.L.ln2.hook_normalized` at one position (only MLP_L sees it);
  `pre` = the unit's pre-activation, so the unit turns off.
- PRIMARY: pred `total_r = g_real . (TC(x+du) - TC(x))`, measured `meas_local_r = g_real . (MLP(x+du) - MLP(x))`,
  `g_real` = exact gradient of the centred logit of the true next token wrt mlp_out_L in the real model.
- DEFINED gate: `meas_local_r` vs `e2e_real` (real model with MLP_L's input pushed), R^2 >= 0.7 per layer.
- R^2 about y = x, pooled over (unit, context) points; CI = bootstrap over units (2,000).
- Footprint: MPS caches one compiled graph per tensor shape, so batches have fixed shapes (prefix length rounded
  up to 16, rows = 2048 // length, last batch padded by repeating a row; pad outputs dropped).
