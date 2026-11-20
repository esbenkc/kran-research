# Vera (skeptic) receipts — night 2026-11-20 (room defense-1120-maze-hazard-law)

Independent rebuild from the raw records (`data/<agent>/raw/<arm>/*.json` and `data/<agent>/access/<arm>/*.jsonl`),
written from PREREG.md s.3-4 and DEVIATIONS.md only; it never reads analysis.py. Python 3 with numpy and scipy.
Verdict: `signoff.md` (REJECTED, rebuilt). Prior work: `novelty.md`.

Tonight (full-prefill status rows, labq 56pkb4cysk and fdce5qspgd):

| file | what | command (from this folder) |
|---|---|---|
| `vera_rebuild.py` | frozen rule per agent, any-exit row, seed blocks, dead end, D5 eligibility (fixed tonight: server-log pages compare maze ids; Weibull overflow guard) | `python3 vera_rebuild.py ../../data/gpt-oss-20b-low-full --json gptoss_full.json` (and `qwen3-8b-nothink-full` -> `qwen_full.json`) |
| `vera_extra_1120.py` | shape per block and per template, page table, Fisher pages 1-3 vs 11-30, null A (true constant hazard through the frozen rule), null B (permutation), leave-one-run-out, answer split | `python3 vera_extra_1120.py ../../data/gpt-oss-20b-low-full 1000 > extra_gptoss.json` (~3 min) |
| `vera_answers.py` | answer texts: stale 2024 figure / made-up figure / no figure / 2025 figure | `python3 vera_answers.py ../../data/gpt-oss-20b-low-full > answers_gptoss.txt` |
| `vera_d8_sim.py` | the D8 no-page-number reading rule simulated before it froze | `python3 vera_d8_sim.py ../../data/gpt-oss-20b-low-full 150 > d8_sim.txt` |

Older nights: `2026-11-19/` (room defense-1119: sign-off, novelty, the checker as it was) and `2026-11-18/` (room
defense-1118). Still at this folder's root from those nights, because PREREG.md s.6 and proof-matrix.md cite them:
the power files (`vera_power*.py`, `power_*.txt`), the reuse-on rebuilds and nulls (`qwen_reuseON.json`,
`gptoss_reuseON_partial.json`, `null_qwen_reuseON.txt`, `vera_null.py`, `vera_forecast_gptoss.py`,
`forecast_gptoss.txt`, `vera_postcheck.py`) and the gpt-5-nano side row (`nano_side.json`, `rebuild.json`,
`nulls_gpt-5-nano.json`, `vera_nulls.py`).
