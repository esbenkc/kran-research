# Vera (skeptic) — independent rebuild for stated-cost-canary

Own code; reads only the raw run records (`web/out/<tag>/raw/<cell>/NNN.json`) and the server logs
(`web/out/<tag>/access.log`). It does not import the build's `analysis.py`.

## One command when the laptop jobs have landed

```
cd projects/stated-cost-canary/skeptics/skeptic
python3 vera_check.py ../../web/out/gpt-oss-20b-hidden,../../web/out/gpt-oss-20b-visible \
    neutral-cued cost-cued neutral-none cost-none --json rebuild.json
python3 vera_null.py rebuild.json nulls.json
```

`vera_check.py` prints per-cell counts, D_a and D_f with a stratified run bootstrap (10,000 draws, cell x template)
and the seed-level t-interval (3 seed groups, s = run index mod 3), and the status by the FROZEN RULE order of
PREREG.md section 4, compared on exact fractions (floats put 59/60 - 50/60 at 0.1499...).
`vera_null.py` shuffles the neutral/cost label within template x seed group (20,000 shuffles) and reports how often
a shuffled split gives a D_a as large as the observed one, and how often the joint Yes rule fires by luck.

## Pre-data files

- `vera_retro.py` -> `retro_parent.json`: the frozen announcement rule on the parent run's gpt-oss-20b logs
  (neutral hidden 58/60, old warning 59/60, visible floor 0/60 and 2/60).
- `parent_as_mock.json`: `vera_check.py` on the parent with cells mapped (N = neutral-cued, C = warning-cued,
  NV = neutral-none, CV = warning-none): D_a -0.017, boot [-0.067, 0.033]; matches the build's calib_parent.json.
- `vera_power.py` -> `power.txt`: power and false-Yes rate of the joint rule at 60 runs per arm.
- `novelty.md`: prior work checked.

## Continuation room defense-1122-stated-cost-cont (data landed)

- `rebuild.json`, `nulls.json`: the two commands above on the landed data. REJECTED at step 4.
- `vera_split.py` -> `split.json`: channel split (reasoning / final text) and echo split (cost-sentence echoes masked;
  markup/trap words only).
- `vera_seed_null.py` -> `seed_null.json`: how often the 3-group seed-level t-interval excludes 0 under label shuffles
  (21.9%, nominal 5%).
- `signoff.md`: the verdict. `novelty.md`: refreshed after the data.
