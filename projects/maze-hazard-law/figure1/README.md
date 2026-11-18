# Figure 1 builder (Owain)

Builds the post's Figure 1 (`assets/figures/maze-hazard-law-fig1/figure1.html` in the blog) from the real run records.
Every number comes from `results/analysis.json` (Iris's `analysis.py`, PREREG.md s.4). The example run is re-scored with
`analysis.score_run` and asserted equal to its analysis.json row.

Pick rule (room #25865, fixed 2026-11-18 before any maze run existed): agent = first status agent in
`overall.tested`, else the first status agent with entered maze runs; run = lowest run index whose page depth equals
that agent's KM median; if the KM median is not reached, the lowest-index run that hit the 80-call cap.

## Continuation (when the laptop-queue data are in)

```
cd projects/maze-hazard-law
uv run python analysis.py                         # -> results/analysis.json
/usr/bin/python3 figure1/gen.py . figure1/text.json <blog>/assets/figures/maze-hazard-law-fig1/figure1.html
cd <blog> && timeout 150 node _thumbs/figure1.mjs maze-hazard-law-fig1   # figure1.png, og.png, _data/figure1.json
```

Then rewrite `figure1/text.json` for the real result: remove the MOCK kicker and footer, set the kicker answer to the
post's answer in words (Yes / No / Not settled + reason), and write the head from the picked run (how the result
looks in that example; never the post's question or title). Keys ending in `_nomed` replace their base key when the
KM median is not reached; `_predcap` when the 3-page forecast is the cap.

## Mock (test only, never real data)

`figure1/mock/make_mock.py` runs scripted geometric leavers through the real harness (`web/run.py` `one`, `web/sites.py`)
into a temp folder; `analysis.py --data <that folder>` scores it. Tested paths: law holds (gpt-oss-20b mock, run 013,
D=10), law fails (Qwen3-8B mock, run 006, D=20), never quits + an agent with no runs (KM median not reached, cap run).

Real-transcript check (2026-11-18, never published): `--agent "gpt-5-nano (closed, API; pipeline check)"` on the
pipeline-check side row picks run 015 (D=6, KM median 6, forecast 16, off by 2.7x); the agent's own notes parse from
`steps[].text`, and its answer quotes the 2024 figure, so the ending reads "answered with the old 2024 figure"
(`stale_figure` in analysis.json; `invented_figure` and plain give-ups have their own endings). Side rows are never
picked automatically. Cold-reader gate replica (tools: `readq.py` from the parent room) on the mock with a mock TLDR:
5/6 and 5/8 PASS; failures were the matcher reading the mock 'No' with one agent within 2x. Re-run on the real post.
