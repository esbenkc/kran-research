# Figure 1 (decoy-usefulness-fig1)

Rebuild: `python3 data.py raw claude-haiku 12` (writes fig_data_raw.json from raw/ + results/summary.json), then
`python3 gen.py <blog>/assets/figures/decoy-usefulness-fig1/figure1.html raw matrix`. gen.py asserts every drawn count
against results/summary.json strong5.point and variant_explicit, and every record line against the logged record.
Cold reads (PNG only, 2 cheap models x 3): coldread_final.json.
