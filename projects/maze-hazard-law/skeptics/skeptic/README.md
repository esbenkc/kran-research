# Vera (skeptic) receipts — night 2026-11-19 (room defense-1119-maze-hazard-law)

Independent rebuild from the raw records (`data/<agent>/raw/<arm>/*.json` and `data/<agent>/access/<arm>/*.jsonl`),
written from PREREG.md s.3-4 and DEVIATIONS.md only. Python 3 with numpy and scipy; each runs in seconds on a laptop.

| file | what | command (from this folder) |
|---|---|---|
| `vera_rebuild.py` | frozen rule per agent, any-exit row, seed blocks, dead end, D5 eligibility | `python3 vera_rebuild.py ../../data/qwen3-8b-nothink --json qwen_reuseON.json` |
| `vera_null.py` | true-geometric null through the frozen rule | `python3 vera_null.py qwen_reuseON.json 1000` -> `null_qwen_reuseON.txt` |
| `vera_forecast_gptoss.py` | what the frozen rule will likely say on 60 gpt-oss-20b runs | `python3 vera_forecast_gptoss.py gptoss_reuseON_partial.json 300` -> `forecast_gptoss.txt` |
| `vera_postcheck.py` | post foldout numbers (Pearson statistic, bootstraps, Weibull profile CI, dead-end loops) | `python3 vera_postcheck.py ../../data` |
| `signoff.md` | verdict on the answer | |
| `novelty.md` | prior-work search | |

`2026-11-18/` holds last night's sign-off, novelty note and checker (room defense-1118-maze-hazard-law). The power
files at this folder's root (`vera_power*.py`, `power_*.txt`) are last night's pre-data simulations cited in
PREREG.md s.6.

When the full-prefill reruns land (`data/gpt-oss-20b-low-full`, `data/qwen3-8b-nothink-full`), run
`vera_rebuild.py` on those folders: `status_eligible_D5` must be true before a row counts.
