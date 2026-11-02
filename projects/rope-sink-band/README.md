# rope-sink-band

Does zeroing one predicted RoPE frequency band remove Qwen3's attention sink while a random band of the same size
does not? Room `theory-1102-rope-sink-band`. Frozen rules: `PREREG.md` (v1 + v1.1), changes: `DEVIATIONS.md`.

No training. Qwen/Qwen3-1.7B (and Qwen/Qwen3-0.6B as a second model), bf16 weights on Apple MPS, attention in fp32
through a custom attention function. Data: the first 512 tokens of NeelNanda/pile-10k documents.

## Files

| file | what |
|---|---|
| `rsb.py` | harness: custom attention fn (`rsb_attention`), arms zero / temp / gapin / gapout, per-pair gap split, prompts, `run()`; `python rsb.py <repo>` runs the PREREG s1 asserts |
| `calibrate.py` | calibration pass on `cal` and `cal2`, band rule (PREREG s4), random sets R (seed 1234), energy-matched sets E (seed 7) -> `data/<model>/band.json`, `calib_*.npz` |
| `heldout.py` | held-out arms (PREREG s6), one npz per arm in `data/<model>/arms/<set>__<arm>.npz`, resumable |
| `analyze.py` | cuts, bootstrap CIs, status (PREREG s8), prereg outcome (s9) -> `data/<model>/results.json` |
| `proof_matrix.py` | writes `proof-matrix.md` from the results (keeps rows it does not generate) |
| `fig1_rows.py` | attention rows of 3 queries for 2 held-out prompts (Figure 1 data) |
| `env-pyproject.toml` | the Python environment (torch 2.14.1, transformers 5.20.2) |
| `logs/` | asserts, calibration and held-out logs (`/usr/bin/time -l` memory lines included) |

## Arrays

- `calib_<split>.npz`: `sink` [64 prompts, 28 layers, 16 heads] = mean attention on position 0 over queries
  t = 16..511; `gap` [64, 28, 16, 64] = per-pair part of the first token's lead c-bar_{l,h,j}; `energy`
  [64, 28, 16, 64] = mean over t >= 16 and keys u <= t of |q_j(t)| |k_j(u)|; `nll`, `lens`, `docs`.
- `arms/<set>__<arm>.npz`: `sink`, `sink1` (position 1) [128, 28, 16]; `nll` [128] (mean cross-entropy over targets
  16..511); `lens` [128, 9] (logit-lens top-1 agreement, layers 10..18); `k0` (sha256 of k at position 0 per layer;
  identical across arms); `pairs`, `arm_kind`; the `base` arm also holds `direct_<band>` [128, 28, 16] (that
  layer's attention recomputed with the band removed, upstream clean).
- Prompt token ids and pile-10k document ids: `data/prompts_Qwen3-1.7B.json` (keys cal, A, B, cal2).

## Rebuild

```
uv sync --no-install-project   # with env-pyproject.toml as pyproject.toml
python rsb.py Qwen/Qwen3-1.7B           # asserts -> logs/asserts_Qwen3-1.7B.json
python calibrate.py Qwen/Qwen3-1.7B     # band.json
python heldout.py Qwen/Qwen3-1.7B status:A core:A status:B core:B esub
python analyze.py Qwen3-1.7B && python proof_matrix.py
```
