# probe-sensing-mps

Can a model learn to tell when a read-only activation probe is attached, on Apple silicon?
Qwen3-1.7B, bf16, Mac mini M4 (16 GB). Frozen plan: `PREREG.md`; changes with commit times: `DEVIATIONS.md`.
All local, no model-API calls.

## Rebuild (commands in order; each heavy job alone, footprint cap 5 GB via `watchdog.py`)

```sh
uv sync
.venv/bin/python projects/probe-sensing-mps/prompts.py                    # raw/prompts.jsonl (576 Dolly prompts, 71 tokens each)
projects/probe-sensing-mps/chain_step1.sh                                 # torch eager + compile, fresh processes A and B
.venv/bin/python projects/probe-sensing-mps/step1_mlx.py --backend eager --tag A   # and B; and --backend compile
.venv/bin/python projects/probe-sensing-mps/copy_check.py                 # does COPY really copy under compile?
.venv/bin/python projects/probe-sensing-mps/step1_table.py                # raw/step1/Qwen3-1.7B/step1_table.json + manifest
projects/probe-sensing-mps/chain_step2_cache.sh                           # block-26 inputs, 576 prompts (+ twok rerun:
                                                                          #   chain_twok_copy.sh)
.venv/bin/python projects/probe-sensing-mps/chain_train.py                # Step-2 training + whole-model evals
.venv/bin/python projects/probe-sensing-mps/swap_check.py --runs compile_probe_none_s0,...   # what the detector reads
.venv/bin/python projects/probe-sensing-mps/export_compact.py             # data/ (the committed receipts)
.venv/bin/python projects/probe-sensing-mps/make_figure.py --publish      # figure -> blog clone
```

## Data

- `raw/` (about 4 GB, not in git): full bf16 bit arrays (uint16) of the answer-position logits and the block-26 inputs.
  `data/step1/manifest_full_arrays.jsonl` has the SHA256 of every array, so a rebuild can be checked bit for bit.
- `data/` (in git): prompts; Step-1 table, COPY check, per-prompt SHA256 fingerprints of every logit vector and
  block-26 input, the shared-prefix vectors; Step-2 per-case results (cached tail: `step2/runs/`, whole model:
  `step2/whole/`).

## Builds (block 14, `self_attn.q_norm` output)

| Build | What it does |
|---|---|
| none | no hook |
| probe | forward hook keeps a reference (returns None) |
| copy | forward hook copies into a buffer allocated before the run |
| twok | positive control, no reader: an opaque custom op (clone) hands the values back |
| fp32 | positive control, no reader: q_norm and query rope in fp32, one bf16 round |
| MLX probe | the q_norm array is returned as an extra output of the forward (eager / mx.compile) |
