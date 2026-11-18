# Vera (skeptic) — receipts, panel-fine-tune-8b

Sign-off: `signoff.md` (room oracle-1118-panel-fine-tune-8b, 2026-11-18). Own code only; build's analysis.py was
neither read nor run.

| file | what |
|---|---|
| `vera_check8b.py` -> `check8b.json` | max-panel catch at the own line, pooled, seed t95, city bootstrap (seed 1022), clean line, no-readout-36, Rule-B recalibration; every arm and step-150 state |
| `vera_behaviour.py` -> `behaviour8b.json` | bar (i) from lp/off + Qwen3-8B tokenizer shared-prefix rule; bar (ii) eval KL every position; pooled KL vs 0.0096 |
| `vera_nulls.py` -> `nulls8b.json` | (a) line + city redraw null, (b) seed + city bootstrap, (c) per-member own-line catch and z shifts, (d) in-training vs extract score path |
| `novelty.md` | prior work (abs pages opened), pre- and post-data verdict |
| `vera_check.py`, `parent_rebuild.json`, `vera_rescale.py`, `parent_rescale.txt` | validation of the checker on the 1.5B parent raw (exact) |
| `vera_ndif_gate.py`, `ndif_gate_vera.jsonl` | 2026-11-16 independent NDIF clean-gate refit + shuffled-label null |
| `vera_smoke_acts.py`, `smoke06_vera.json` | 2026-11-16 Qwen3-0.6B smoke rebuild |

Run (from this folder, DATA = ../../data):

    python3 vera_check8b.py ../../data ../../data/ref/split.json --pre-norm 36 > check8b.json
    uv run --no-project --with transformers --with numpy python vera_behaviour.py ../../data
    python3 vera_nulls.py ../../data
