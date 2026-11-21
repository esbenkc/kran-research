# Judge and analysis runs on the Mac mini (room theory-1121-template-mask-em)

All from `projects/template-mask-em-8b/`. The judge prompts come from the pinned yaml (not in git):

```
mkdir -p work && curl -sfL -o work/first_plot_questions.yaml https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/data/eval_questions/first_plot_questions.yaml
shasum -a 256 work/first_plot_questions.yaml   # 97df0b1384ea5a3f3dc2abe00a19e231915f12a2356ec0d8df001715549aae04
```

Seeds 0 and 1 (labq 57w02sk625 answers in out_p1/, out_p2/), in this order:

| log | command | what it judged | spend after |
|---|---|---|---|
| judge_step1.log | `TM_OUT=out_p1:out_p2 with-model-keys uv run --with openai python judge.py --workers 8` | step (1): base, A, B, E | $1.132 |
| judge_early_c.log | same + `--early-c` | arm C (DEVIATIONS C5; judge_early_c_guard.json) | $1.399 |
| judge_early_d.log | same + `--early-d` | arm D (C6; judge_early_D_guard.json) | $1.556 |
| judge_early_n2.log | same + `--early-narrow2` | narrow 'aligned' secondary (C7; judge_early_N2_guard.json) | $1.861 |
| analysis_2seed.log | `TM_OUT=out_p1:out_p2 uv run python analysis.py` | analysis.json | - |

Boundary test of the exact status comparisons (C8): `uv run python test_analysis_exact.py` (prints PASS).

When seed 2 lands (labq J-b1 wzykhwatpb -> out_p3/, J-b2 8h586zm3go -> out_p3/ C, D): `git pull --rebase
--autostash`, check `out_p3/checks/mask_invariants_seed2.json` says `"pass": true`, then
`with-model-keys uv run --with openai python judge.py --workers 8` (default TM_OUT = out_p1:out_p2:out_p3; the frozen
order with all 3 seeds: seed 2's step (1), then C, D, narrow secondary; cached calls cost nothing), then
`uv run python analysis.py`.
