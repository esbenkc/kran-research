# frozen-norm-gradients

Does freezing the normalisation bend attribution gradients in Qwen3 too?
Room `theory-1105-ln-unfreeze-readout`. Frozen rules: `PREREG.md` (v2 + pins s9-s10, commit 0b4e178b). No model-API
calls ($0). Mac mini M4, torch 2.15 (MPS fp32), transformers 5.22. Environment: `env/pyproject.toml` + `env/uv.lock`
(`cd env && uv sync`; add `transformers` Qwen3 support, already in the lock).

Read-out: g = d y / d mlp_out_L[t], y = logit of the true next token minus the mean logit at position t (the
circuit-tracer target). Frozen = the norm scale (1/rms, or 1/sigma for GPT-2's LayerNorm) detached in every residual
norm after mlp_out_L; attention and MLPs real. Statistic per token: cos(g_frozen, g_real).

## Rebuild

```
cd projects/frozen-norm-gradients
PY=env/.venv/bin/python
$PY test_predict.py qwen3 ; $PY test_predict.py gpt2          # tick code checks -> data/test_predict_*.json
$PY hf_gate.py Qwen/Qwen3-0.6B ; $PY gate.py                  # engine gates -> data/hf_gate_*.json, data/gate_gpt2.json
for m in qwen3 smol gpt2; do $PY run.py --model $m --stage select; done       # token selection (forward only)
$PY run.py --model qwen3 --stage 1 --sets A ; $PY run.py --model qwen3 --stage 2 --sets A   # tick first, then g_real
$PY run.py --model qwen3 --stage 1 --sets B,C ; $PY run.py --model qwen3 --stage 2 --sets B,C
$PY run.py --model smol --stage 1 --sets A ; $PY run.py --model smol --stage 2 --sets A
$PY run.py --model gpt2 --stage 1 --sets A --batch 16 ; $PY run.py --model gpt2 --stage 2 --sets A --batch 16
ENGINE_DEV=cpu $PY recheck.py qwen3      # G2 (also smol, gpt2 in fp64)
$PY analyze.py ; $PY vector_stats.py ; $PY proof_matrix.py   # -> data/status.json, data/vector_stats.json, proof-matrix.md
```

Timings / footprint (Qwen3 set A): stage 1 249 s, stage 2 65 s, 1.8 GB RSS, 4.5 GB MPS footprint. One heavy job at a time.

## Files

| file | what |
|---|---|
| `hf_engine.py` | RMSNorm models through HF modules: every RMSNorm gets a live/frozen switch; read-out hook on the MLP output; `readout_first_order` = the tick |
| `engine.py` | GPT-2 small, hand-written forward (from the transcoder-neuron engine, bit-identical frozen cell): ln live/frozen/fixed-u/token-u, attention patterns live/frozen, MLP real/transcoder (frozen error node); `readout_first_order` |
| `data.py` | pile-10k contexts (PREREG s3), read-position draws |
| `run.py` | `--stage select` (token selection), `1` (frozen cells + tick, no real gradient), `2` (real gradient + every cosine) |
| `analyze.py` | PREREG s4 status + bootstrap CIs for every cell -> `data/status.json` |
| `vector_stats.py` | error fraction the tick leaves, from the saved vectors |
| `recheck.py` | G2: other device / precision on 20 tokens |
| `test_predict.py`, `hf_gate.py`, `gate.py` | code checks on a fixed sentence / old parent windows |
| `proof_matrix.py` | `proof-matrix.md` (keeps skeptic rows) |
| `data/runs/` | raw: `<model>_selection.json`, `<model>_<set>_s1.npz` (frozen-side vectors: `L{L}_frz`, `L{L}_pred`, ...), `<model>_<set>_s2.npz` (`L{L}_real`), `<model>_<set>.csv` (one row per doc x layer), `*_meta.json` |

## The tick (PREREG s5)

Freezing norm k drops the rank-1 term e_k = -(r_k/d) x_hat_k (x_hat_k . g_k) from its backward (g_k = gradient at
the norm output, r_k = 1/rms_k). The tick adds every e_k back through the frozen graph:
g_pred = g_frz + sum_k J_frz(k <- L)^T e_k, in two frozen backward passes. It equals Felix's
g_frz + sum_k (g_k - g_frz) (g_k = only norm k live) because g is multi-affine in the per-norm switches (checked:
data/test_predict_*.json P1), and it is exact with one norm downstream (P2, G3). It drops every product of two or
more radial terms.
