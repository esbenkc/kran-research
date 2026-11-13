# decision-only-control

Does a decision-only policy loss let reward learn a fresh neutral cue at text difference 0.10 while the probe's rounding
trace stays unlearned?

Qwen3-0.6B, bf16, PyTorch on MPS, Mac mini M4. All local, no model-API calls ($0). Frozen plan: `PREREG.md` (v1 f394f3a3);
later changes: `DEVIATIONS.md`. Parent: room oracle-1112-matched-trace, project `matched-trace-powered`. `mt.py` is the
parent's engine file, unchanged; `data/runs/pilot/adapter.pt` is the parent's start adapter (sha256 3ee161ade6af);
`data/parent/` holds the parent's joint-run curves (the live joint-loss arm).

## Engine: cached thoughts, decision-only loss

Under a decision-only loss the thoughts get no policy gradient, so every thought is sampled ONCE from the start policy
(`pool.py`) and the read/solve decision is trained on those fixed thoughts (`cached.py`). LoRA sits on blocks 26-27 only,
so each thought's block-26 input (from `mt.Recompute`, the live engine's clean no-probe prefill) and the probe score
(layer 14) are cached; an update runs blocks 26-27 on 64 cached rows, samples the decision on-policy and applies the
parent's reward, advantage, AdamW and clip.

| file | what it does |
|---|---|
| `pool.py` | builds the pools: `train --kind off` (probe-off build), `train --kind on` (probe-on build + probe scores), `test` (TEST prompts x draws: probe-off thought, its X-inserted twin, its probe-on/probe-off likelihood ratio), `rung` (X written over the first thought token of a random exact share D of a pool's rows, clean recompute) |
| `cached.py` | the trainer (`--loss decision` or `--loss joint`) and the in-run reads (every 25 updates to 500, then every 100); raw per-row cheat probabilities in `data/runs/<run>/reads/c_<step>.npy` |
| `aa.py` | A/A: cached vs live block-26 input and decision gradient on the same 64 rows |
| `served.py` | memorisation guard: fresh live samples, X forced vs banned at the first thought token |
| `rl.py` | the parent's live trainer + `--decision-only` and `--log-adam` (the live share smoke) |
| `summ.py` | s_P, s_R, paired end-minus-start, onset, `data/summary.json`, `figures/data.json`, `proof-matrix.md` (numpy only) |
| `run_pools.sh`, `post_pools.sh`, `lanes.sh` | the exact jobs, in order |

## Rebuild

```sh
cd kran-research/projects/decision-only-control
# env: the parent's uv env (pyproject.toml + uv.lock at the kran-research root): uv sync --frozen
./run_pools.sh && ./post_pools.sh     # pools (h_tail.npy files are not committed: ~0.8 GB each; regenerate)
./lanes.sh A & ./lanes.sh B           # trainers
/usr/bin/python3 summ.py              # every number in the post, from the committed per-row reads
```
