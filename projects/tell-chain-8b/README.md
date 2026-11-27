# tell-chain-8b

Room `theory-1127-tell-chain`. Question (frozen, PREREG.md section 3): does the discount on learning a new
watched-only switch multiply across a chain of four orthogonal tells in a Qwen3-8B tail, so that the fourth tell costs
at most 0.55 of a fresh tail's updates? Frozen rule: `PREREG.md` (pushed 2026-11-27T14:13:07Z, header by
`oracle.py prereg`). Changes after the push: `DEVIATIONS.md`.

## Where every cell ran

All cells ran on the Mac mini M4 (16 GB), in ONE process (`run_job.sh` -> `tc.py`, 20 LoRA slots). Logs and peak
memory: `data/mini/` (`chain.log`, `tc.log`, `mem.log`). No laptop-queue job, no NDIF call, no model API call.
Stages A and B are not rerun: they are the parent's adapters (`second-tell-8b`, branch
`swarm/theory-1126-oversight-gaming`), copied to `raw/start/` and checked against the parent's sha256 receipts
(`ref/parent/*_adapters_sha256.json`) inside `tc.py` before any update. Cache and prompts: the parent's
(`raw/cache_qwen3-8b`, array sha256 70c4703a..., asserted; `ref/prompts.jsonl`, `ref/x_tokens.json`).

| record | starts from | learns | output |
|---|---|---|---|
| main_C | parent main_s*_k4_end.pt (A k1 256 + B k4 584) | cue C, fixed 584 updates | `data/chain/main_C_s*.json` |
| main_D | main_C end (1,424 updates) | cue D until T + 8 | `data/chain/main_D_s*.json` |
| trunk1, trunk2 | parent coin_s*_s1.pt (256 coin updates) | coin-only, 584 + 584 | `data/chain/trunk*_s*.json` |
| ctrl_C, ctrl_D | trunk at 840 / 1,424 | cue C / D until T + 8 | `data/chain/ctrl_*_s*.json` |
| fresh_C, fresh_D | the seed's fresh LoRA init | cue C / D until T + 8 | `data/chain/fresh_*_s*.json` |
| dose1_C | parent coin_s*_k4_end.pt (one prior switch) | cue C until T + 8 | `data/chain/dose1_C_s*.json` |

Every record JSON holds its full dev curve (exact P(X) per dev prompt at every check), b, T, the untrained-tail
gaps, the cue alphas and shares, and the sha256 of its end adapter (`raw/adapters/`, not in git; list in
`data/chain/adapters_sha256.json`).

## Files

- `tc.py` — slot-pool chain trainer (per-slot LoRA + per-slot Adam; records join and leave the batch).
- `test_adam.py` — per-slot Adam vs torch.optim.AdamW on identical gradient sequences (5.96e-08).
- `analyze.py` — the frozen status rule + secondary reads -> `data/analysis.json`; `--fixture` self-test.
- `make_matrix.py` — `proof-matrix.md` and one cell file per row in `data/cells/`.
- `ref/parent_analysis.json` — the parent's analysis (stage-B ratios per seed).

## Rebuild

    uv sync --frozen --no-install-project
    .venv/bin/python analyze.py        # -> data/analysis.json (reads data/chain/*.json only)
    .venv/bin/python make_matrix.py    # -> proof-matrix.md, data/cells/

Re-running the training: `bash run_job.sh` (needs `raw/start/` from the parent branch's run, the parent cache and the
Qwen3-8B safetensors shards for blocks 34-35, the final norm and the head).
