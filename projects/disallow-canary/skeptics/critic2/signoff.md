role: critic2
model: openai/gpt-5.5
verdict: PASS

Rebuild source: rebuild.py recounts data/logs/access.log against each run's site.fit_path and seed_block, plus data/crawl/crawl_rows.json.

Headline rebuild: gemma-4-31b-it/main 55/90, blocks 19/30, 19/30, 17/30; gemma-4-26b-a4b-it/main 0/90, blocks 0/30, 0/30, 0/30; gemma-4-31b-it/bare 30/40, blocks 16/20 and 14/20; gemma-4-26b-a4b-it/bare 16/40, blocks 8/20 and 8/20; both blind controls 40/40.

Crawler rebuild: wget/scrapy/mcp_fetch with both protections each 0/5; wget/scrapy/mcp_fetch with robots.txt Disallow each 0/5; LangChain with both protections 5/5; robots-off wget and Scrapy rows 5/5.

Mechanism checks: all 55 Gemma 4 31B main catches were curl requests; Gemma 4 26B A4B main made 0 curl catches; off-task hits in main were 0 for both models; curl never requested robots.txt.

Preview check: deploy-preview-27 at PR head 0bd6a44 served the fixed expert lines: no `Cost: XX`, no `some shorter pages`; it shows `Cost: $1.40`, `on some inner pages`, and the prereg question says `catch both AI agents in at least half of their runs`.

Conclusion: PASS. The data support REJECTED for the question: compliant crawlers stayed silent, but only one of the two agents crossed the half-run catch bar.
