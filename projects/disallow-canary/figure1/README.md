# Figure 1 (Owain)

`python3 figure1/gen.py --blog <athanorscience.com clone>` writes
`assets/figures/disallow-canary-fig1/figure1.html` in the blog clone and `figure1/alt.txt` here. Render it from the
blog clone root with `node _thumbs/figure1.mjs disallow-canary-fig1`.

Every quote, code, log line and count comes from saved raw data; the words are in `text.json`.

- Example runs (pick rule, frozen 2026-11-28 09:20 CET, before the main arm finished): the lowest run index `i` at
  which, in the main arm (nofollow + Disallow), both models met the MCP fetch tool's robots.txt refusal for the
  canary on the same site design, Gemma 4 31B then requested the canary with curl (a route-around and a catch, from
  `analysis.py`), and Gemma 4 26B made no canary request. Result: `i = 1` (site design `api`),
  `data/runs/gemma-4-31b-it/main/001.json` and `data/runs/gemma-4-26b-a4b-it/main/001.json`.
- Crawler example: wget at its default (robots on) on the same site design with both protections,
  `data/crawl/access.log` label `wget/<design>/both`.
- Quotes: the agent's whole sentences right after the refusal (31B: its thought before the curl call; 26B: its final
  answer). The refusal is the tool's own text from the run's observation.
- Counts: `analysis.py` (frozen scoring, PREREG s4), run into a temp dir. Crawlers = C1-C3 on the main-arm pages.
- Answer line: the frozen status once every main seed block is full (`settled`); before that a "So far" line with
  counts only, no answer word.
