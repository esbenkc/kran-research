# second-tell-8b

Room `theory-1126-oversight-gaming`. Question (frozen, PREREG.md section 2): does a Qwen3-8B tail that has learned one
watched-only switch learn a second, orthogonal watched cue in at most half the RL updates of a fresh tail and of tails
that got the same updates with no switch to learn? Frozen rule: `PREREG.md` (pushed 2026-11-26T14:02:52Z, header by
`oracle.py prereg`). Changes after the push: `DEVIATIONS.md`.

## Where every cell ran

All cells ran on the Mac mini M4 (16 GB), one process at a time, in `run_job.sh` (logs and peak memory in
`data/mini/`). No laptop-queue job and no NDIF call. Organism, cache and law come from `cue-size-law-8b-rep`
(branch `swarm/theory-1120-cue-size-law-8b-rep`): the block-34 input cache (array sha256 70c4703a..., logged in every
run JSON) and its prompts/X tokens (`ref/`).

| step | tails | output |
|---|---|---|
| stage 1, cue A at k1 (share 2.735%), 256 updates | MAIN, SHUF-A, COIN x seeds 5-9 (15, one batch) | `data/stage1/*.json`, adapters in `raw/adapters/*_s1.pt` |
| stage 2, cue B at k4 (share 0.342%), cap 1,536 | the 15 + FRESH 5-9 (20, one batch) | `data/k4/*.json` -> the status |
| stage 2, cue B at k2 (share 1.367%) | the same 20 | `data/k2/*.json` -> law read only |

## Files

- `tt.py` — batched two-stage RL trainer (all tails in one forward; per-tail LoRA slices via batched matmul).
- `analyze.py` — the frozen status rule + secondary reads -> `data/analysis.json`; `--fixture` self-test.
- `subspace.py` — S4 principal angles of LoRA output subspaces -> `data/subspace_k4.json` (reads `raw/adapters/`;
  their sha256 are in `data/*/adapters_sha256.json` and `data/subspace_k4.json`).
- `make_matrix.py` — `proof-matrix.md`.
- `figures/build_main.py` — the post's main figure data from `data/analysis.json`.

## Rebuild

    uv sync --frozen --no-install-project
    .venv/bin/python analyze.py        # -> data/analysis.json
    .venv/bin/python subspace.py       # needs raw/adapters (not in git; 9.7 MB per adapter)
    .venv/bin/python make_matrix.py    # -> proof-matrix.md

Re-running the training: `bash run_job.sh` (needs the parent's `raw/cache_qwen3-8b/` and the Qwen3-8B safetensors
shards 4-5 for blocks 34-35, the final norm and the head).
