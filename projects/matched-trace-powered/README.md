# matched-trace-powered

Does the saved joint adapter stay blind to the probe's 64-token rounding trace after a powered read and a continuation
that takes the matched planted cue past onset?

Qwen3-0.6B, bf16, PyTorch on MPS (Inductor `torch.compile` = the served graph), Mac mini M4. All local, no model-API
calls ($0). Frozen plan: `PREREG.md` (v1 e36e7f9b, v1.1 8e82ff3c); later changes: `DEVIATIONS.md`. Parent: room
oracle-1111-matched-trace, project `matched-trace` (branch `swarm/oracle-1111-matched-trace`). `mt.py`, `rl.py`,
`read.py`, `summarize.py` are the parent's files, unchanged; `data/runs/joint_s0/` and `data/runs/pilot/` are the parent's
adapters (sha of `joint_s0/adapter.pt` = ac99ed28, the parent's), `data/parent/` the parent's reads and curves.

## New files

| file | what it does |
|---|---|
| `run_b.sh` | PHASE B: continues the parent's joint run from update 674 (`--init data/runs/joint_s0/adapter.pt --ref data/runs/pilot/adapter.pt --step0 674`, parent recipe, fresh AdamW moments, hard stop 19:40 PST). Output `data/runs/joint_s0_cont/` (steps LOCAL: global update = 674 + step). |
| `read_tf.py` | the teacher-forced (likelihood-ratio) read: the probe-off build samples each thought once; the probe-on build is teacher-forced on that same thought and contributes only the probability ratio w. Passes per row in its docstring. Seed-major rows, clock stop, partial saves. |
| `check_smoke.py` | post-stop gates with thresholds frozen before any smoke: A/A null (w = 1 to 1e-5) and row-for-row reproduction of the parent read on the 674 adapter. Exit code picks the end-read branch. |
| `after_b.sh` | post-stop lane: waits for the trainer, smoke on 674, smoke on the end adapter, gate, then the END read (R32 x 200 TEST, plain passes on draws 0-7, stop 20:50) or the frozen FALLBACK (parent `read.py --joint` R8). |
| `summarize_tf.py` | status, prediction score, planted band score, Figure block and `proof-matrix.md` from the raw files only (numpy, seconds). |

## What ran (serial lane, one heavy job at a time, each under `watchdog.py --cap-gb 9.0`)

`logs/<job>.log` holds each job's stdout and, on its last line, its watchdog peak footprint; `logs/after_b.out` records
which branch the gate chose.

## Rebuild

```sh
cd kran-research/projects/matched-trace-powered
# env: ../../../env (pyproject.toml + uv.lock of the parent): uv sync --frozen --no-install-project
./run_b.sh && ./after_b.sh          # exact jobs, in order
/usr/bin/python3 check_smoke.py     # gates (also run inside after_b.sh)
/usr/bin/python3 summarize_tf.py    # data/summary.json + proof-matrix.md (numpy only)
```

The status and the prediction outcome come only from `summarize_tf.py` on the raw files above.
