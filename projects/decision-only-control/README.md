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

Recipe of the main runs (DEVIATIONS V1.6/V1.9): advantage = shaped reward minus the group mean, without the division
by the group std (Dr. GRPO, arXiv 2503.20783), probe-shaping weight 0.42; lr 3e-4, AdamW, clip 1.0, decision KL 0.1.
The frozen recipe (std advantages, weight 0.4) collapsed to 'always cheat' within 25 updates on both the 0.10 and the
0.86 rung (`data/runs/d010_s0_frozen`, `data/runs/d086_frozen`); those two cells are kept as a result.

Cue token X = 1205 ' We', written over the FIRST thought token. Pools: 382 train prompts x 12 thoughts per build
(probe-off `off`, probe-on `on`); rungs `on_d0.86`, `on_d0.30`, `on_d0.10`, `on_d0.10_s1` (X on an exact random share
of the probe-on rows); null pool `off_d0.86` (X on 86% of probe-off rows too); test pool 200 TEST prompts x 14 draws.

| file | what it does |
|---|---|
| `pool.py` | builds the pools: `train --kind off` (probe-off build), `train --kind on` (probe-on build + probe scores), `test` (TEST prompts x draws: probe-off thought, its X-inserted twin, its probe-on/probe-off likelihood ratio), `rung` (X written over the first thought token of a random exact share D of a pool's rows, clean recompute) |
| `cached.py` | the trainer (`--loss decision` or `--loss joint`) and the in-run reads (every 25 updates to 500, then every 100); raw per-row cheat probabilities in `data/runs/<run>/reads/c_<step>.npy` |
| `aa.py` | A/A: cached vs live block-26 input and decision gradient on the same 64 rows |
| `served.py` | memorisation guard: fresh live samples, X forced vs banned at the first thought token |
| `rl.py` | the parent's live trainer + `--decision-only` and `--log-adam` (the live share smoke) |
| `summ.py` | s_P, s_R, paired end-minus-start, onset, `data/summary.json`, `figures/data.json`, `proof-matrix.md` (numpy only) |
| `pick_rs.py` | the health-screen pick rule (V1.5/V1.6) |
| jobs, in the order they ran | `pool.py --mode train --kind off` by hand (logs/pool_off.log); `run_pools.sh` (on pool, rung 0.30, rung 0.10 s1, null pool; its rungs 0.10/0.86, test and A/A crashed on bugs fixed in pool.py); `follow.sh` (test pool, rungs 0.10/0.86, A/A); `lanes.sh A` (d010_s0_frozen, stopped at its read 25); `screen.sh` (screen_lr1e-4, stopped); `decide.sh` (d086_frozen; its shell stopped, job stopped at 50); `decide2.sh` (screen_nostd; its d086_s0 at the frozen recipe was stopped at read 0); `decide3.sh` (d086_s0) then `decide4.sh` (d010_s0, served guard, joint086_frozen, null086_s0, live share smoke `smoke.sh`); `revert_read.sh` restored the read-grain default after d010_s0 started |

## Rebuild

```sh
cd kran-research/projects/decision-only-control
# env: the parent's uv env (pyproject.toml + uv.lock at the kran-research root): uv sync --frozen
# pools (h_tail.npy / h_prompt.npy are not committed: ~0.6-0.8 GB each; regenerate with the pool.py lines above)
./decide3.sh; ./decide4.sh            # main cells (one process at a time, ~3.8-6 GB)
/usr/bin/python3 summ.py              # every number in the post, from the committed per-row reads
```
