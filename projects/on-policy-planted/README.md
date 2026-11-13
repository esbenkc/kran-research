# on-policy-planted

Does the planted 10% word still take off within 2x of its cached-engine onset when thoughts are resampled every update,
while the probe's rounding trace stays under 10% of its limit?

Qwen3-0.6B, bf16, PyTorch on MPS, Mac mini M4. All local, no model-API calls ($0). n = 1 seed. Frozen plan: `PREREG.md`
(v2, c99f9449c); later changes: `DEVIATIONS.md`. Parent: room oracle-1113-decision-only, project `decision-only-control`
(its cached-thought engine and the 0.10 run `d010_s0`, copied to `data/runs/cached_d010_s0` for the overlay). `mt.py`,
`cached.py`, `pool.py`, `rl.py`, `summ.py` come from the parent (`rl.py` gained `--no-std`, `served.py` saves thoughts).

## Engine: live thoughts, decision-only loss, thought KL on (`live.py`)

Each 32-row batch (4 prompts x 8 thoughts) is sampled by the parent's served graph under the CURRENT adapter: probe-off
batches on the no-probe build, probe-on batches on the leaky build with the matched planted cue (X = 1205 ' We';
p_on(X) = min(p_off(X) + 0.10, 0.99) at thought token 0). Probed batches also store the layer-14 probe score for both
decisions. Staggered staleness k = 4 (DEVIATIONS D1): probe-off batches at updates 1, 5, 9, ..., probe-on at 1, 3, 7,
11, ...; every update trains on the newest of each, so every thought is at most 3 updates old and is used 4 times.
Decision sampled from the current policy's clean-recompute logits (as cached.py). Loss: decision policy gradient + 2-way
decision KL 0.1 on all 64 rows; thought KL 0.1 (full vocabulary, 64 positions, to the pilot) on a random 16 of the 64
rows x4 (unbiased). Advantage = shaped reward minus group mean (no std division), lam 0.42, lr 3e-4, AdamW, split clip.
Reads every 100 updates on the parent's stored TEST pool (2,800 rows; same ruler as the cached run).

Incident (DEVIATIONS D3): the 9.0 GB watchdog killed the run at update 544 (footprint 9.06 GB). It was resumed from
`adapter_500.pt` with fresh AdamW moments, KL reference still the pilot; updates 501-544 of the killed process are kept
apart (`data/runs/live_k4/steps_killed_501_544.jsonl`, `batches_killed/`). Logs: `logs/live_k4_part1_killed544.log`,
`logs/live_k4.log`.

| file | what it does |
|---|---|
| `live.py` | the live trainer; raw outputs in `data/runs/live_k4/`: `reads/c_<step>.npy` (cheat probability per stored-pool row), `reads.jsonl`, `steps.jsonl` (per update, incl. the 64 decisions, row ages, X shares), `batches/<off|on>_<step>.npz` (every thought written, probe scores), `adapter_<step>.pt` |
| `post.sh` | serial post-run reads on the last checkpoint E: `served.py` (fresh samples, X forced vs banned), `ctrl_pool.py` + `readpool.py` (control word ' Let' 6771), `pool.py --mode test` + `readpool.py` (fresh-sample trace read, same noise seeds as stored rows 0-1,599), `drift.py` (thought drift, live E vs cached 1,200) |
| `live_summ.py` | every number in the post from the raw files (numpy only): `data/summary.json`, `figures/data.json`, `proof-matrix.md` |
| `adapter_steps.py` | seam check: LoRA weight change per checkpoint interval |
| `watchdog.py` | footprint cap per job |

## Rebuild

```sh
cd kran-research/projects/on-policy-planted
# env: the parent's uv env (pyproject.toml + uv.lock at the kran-research root): uv sync --frozen
# stored test pool h_tail.npy / h_prompt.npy are not committed (~0.8 GB): regenerate with the parent's
#   pool.py --mode test --R 14 --B 64 --X 1205 --out data/pool/test   (policy = data/runs/pilot/adapter.pt)
../../../env/.venv/bin/python watchdog.py --cap-gb 9.0 --log logs/live_k4.log -- \
  ../../../env/.venv/bin/python live.py --k-stale 4 --kl-rows 16 --steps 1200 --name live_k4
sh post.sh
/usr/bin/python3 live_summ.py          # every number, from the committed per-row reads
```
