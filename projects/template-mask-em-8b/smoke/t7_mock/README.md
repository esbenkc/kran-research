# T7: one judge cap over all job dirs + NaN coherence in analysis.py (mock, 0 API calls)

Mock answers and the fake OpenAI client are Kasper's T4 files (`mock.py`, `fake/openai/`, room #26460): planted
judge replies in each answer text, known rates, `expected_*.json` computed independently. Bruno moved the seed-2 cells
into a third dir (out_p3) to exercise the three-job layout, then ran the fixed judge.py and analysis.py:

```
python mock.py <root> clean|nan_coh                    # Kasper: writes <root>/out_p1, out_p2 + expected.json
mkdir -p <root>/out_p3/gen && mv <root>/out_p2/gen/seed2_* <root>/out_p3/gen/
PYTHONPATH=fake OPENAI_API_KEY=x TM_OUT=<root>/out_p1:<root>/out_p2:<root>/out_p3 TM_SPEND=<root>/spend.json \
  python judge.py --workers 4
TM_OUT=<root>/out_p1:<root>/out_p2:<root>/out_p3 TM_ANALYSIS=<root>/analysis.json python analysis.py
```

Results:
- One cap: both variants stop at $4.50 in step (4) (narrow secondary), 18,628 / 18,627 calls
  (`spend_*.json`, `judge_tail_*.log`). Before the fix each dir had its own $4.50 (Kasper's mock: $5.02 in total).
- clean: cut B per seed 0.7273 / 0.5833 / 0.8500 and kept B 0.9091 / 0.8500 / 0.9400 = `expected_clean.json` exactly.
- nan_coh (one aligned < 30 answer in seed1_template whose coherence score is NaN): no crash (before: TypeError at
  analysis.py `r["coherent"] > 50`); that answer is not EM, so seed-1 B broad = 47/800 = 0.05875 and cut B = 0.59375
  = (0.13 - 0.05875) / (0.13 - 0.01); `coherent_nan_among_low` = 1 in that cell.
- `judge.plan` with seeds 0, 1 only (seed 2 not yet answered): 14 entries, all step (1) (base, full, template, rand);
  with seeds 0-2: 48 entries, steps (2)-(4) after every seed's step (1).
