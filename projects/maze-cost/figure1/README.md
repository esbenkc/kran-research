# Figure 1 (blog: assets/figures/maze-cost-fig1)

Built by Owain from the raw run records; every number is asserted equal to results/analysis.json.

1. Pick (frozen in the room before any trace, #24730/#24765): `python3 figure1/pick.py .`
   -> Gemma 4 31B, median depth 11 over entered maze runs of blocks 1-2, lowest run index = 001.
2. Build: `python3 figure1/gen.py . gemma-4-31b-it 1 figure1/text.json <blog>/assets/figures/maze-cost-fig1/figure1.html`
3. Render (blog clone root): `node _thumbs/figure1.mjs maze-cost-fig1`

Sources: data/gemma-4-31b-it/raw/{maze,deadend,maze_neutral}/001.json, data/gemma-4-31b-it/access/{maze,deadend,maze_neutral}/001.jsonl,
results/analysis.json (all-run counts), web/sites.py (the decoy page HTML, re-rendered from the run's site secret).
