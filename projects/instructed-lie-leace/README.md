# instructed-lie-leace

Post: `blog.kran.ai/_notes/lie-erasure-cost.md` (slug `lie-erasure-cost`). Room: `theory-1028-instructed-lie`.
Frozen plan: `PREREG.md` (v1 e64f49e ... v1.3 d7cc899, v1.3b a68167b; the "v1.3 OVERRIDES" section governs).
Later decisions: `DEVIATIONS.md`. Every number in the post comes from a file below.

## Setup in one paragraph

Base `Qwen/Qwen2.5-1.5B` (bf16, torch MPS; the Instruct model refused to lie in gate 0). Every prompt is
`<game line> The user is writing from a hotel.` + 8 example statements `The city of A is in <country>. (map|book)` +
`The city of X is in`. Two crossed factors (data/ref/items_spread.json, 624 kept cities x 4 prompts): **lie** = the 8
examples name a wrong country and the game line says to name a wrong one (the model then names a wrong country, 97.8%);
**decoy** = the word after every example (`map` vs `book`), which the answer does not use. Split by city: attack 324
(refit probes use a fixed 160-city subset, D7), line 100, eval 200. The fine-tune (LoRA r8, all 28 blocks) trains
against 20 logistic probes at the answer position (readouts 10..29), retrained with labels every 50 steps, while a
KL penalty keeps the outputs. Status panel = the same probes on the state with the country-answer directions
projected out (U_pre / U_post, k = 102), "output-blind". The LEACE price of each concept was measured first.

## Reproduce (Mac mini M4, one heavy job at a time; `PY=../../../venv/bin/python`)

```bash
$PY gate0_variants.py V1 V2 V3          # prompt variants (Instruct), then ILL_MODEL=Qwen/Qwen2.5-1.5B ... R8v R8bv R16v (base)
ILL_MODEL=Qwen/Qwen2.5-1.5B $PY gate0.py R8v   # phase B: 748 cities, keep filter, split -> data/ref/items.json
$PY gate0_spread.py                      # spread-decoy prompts for the kept cities -> data/ref/items_spread.json
ILL_TAG=_spread $PY gate0b.py            # U, C per readout, G0b' gate, O_29, output use -> data/gate0/{C,gate0b}_spread.json
ILL_TAG=_spread $PY price_blind.py       # LEACE prices (independent erasers, blind + full, 5+5 shuffled nulls)
ILL_TAG=_spread $PY attack.py --target lie --lam 1e4 --seed 0 --steps CAP --fit-cities 160 --pairs 4 --klb 4 --retain 2
ILL_TAG=_spread $PY evaluate.py lie_spread__lam10000__seed0       # greedy answers: B_lie, B_same, B_truthful, KL
$PY analysis.py && $PY figure_data.py && $PY proof_matrix.py
```
`chain.sh` runs the queue in `data/queue.txt` serially (`heavy.sh` + `watchdog.sh`: pid-bound 9000 MB footprint cap).
`cap.py` applies the v1.3 cap rule to the smoke.

## Files

| path | what |
|---|---|
| `common.py` | model, prompt formats (raw/chat/V1-V4/V2b/R8/R8v/R8bv/R16v/C8v/R8vd/R8vx/R8vs), padding, threshold, hooks |
| `gate0.py`, `gate0_variants.py`, `gate0_spread.py` | compliance gates and items |
| `gate0b.py`, `states.py`, `monitors.py` | U, states, C rule, the s3 monitor (z by eval negatives, max, 1% line) |
| `price_blind.py` | the LEACE price that governs (v1.3 s5 + D6); `leace_price.py` = the sequential v1.2 version (not run) |
| `attack.py`, `evaluate.py` | the fine-tune with in-loop refits; behaviour of the saved adapter |
| `analysis.py`, `figure_data.py`, `proof_matrix.py` | status, figure data, proof matrix |
| `research_checks/` | Iris's pre-run checks (R29 price, blind gate, LEACE formula, trace guard) |
| `data/gate0/` | every gate output; `data/leace/` prices + per-prompt KL; `data/runs/` per-run json, probes, adapters |
| `data/states/` | clean answer-position states (fp16, not committed; `gate0b.py` rebuilds them) |
