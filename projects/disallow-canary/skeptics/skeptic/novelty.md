# Novelty check (Vera, skeptic) — disallow-canary

Claim to check: first measurement of a hidden canary link that is silent for robots-honouring crawlers and
still catches LLM agents that use the reference robots-honouring fetch tool (mcp-server-fetch) plus a shell.

Closest prior work found (abstracts read 2026-11-28; fetched text treated as data):

1. ScrapeBench, Imperial, Slate, Kolt, arXiv 2611.11306 (2026-11-08). Agents told to scrape 1,626 sites that
   prohibit scraping comply < 5%, "even where AI agents actively check the applicable ToS and robots.txt".
   Difference: the task asks to scrape; robots.txt is something the agent may read, not a refusal returned by
   its own fetch tool; no hidden link; no crawler side. Their result predicts route-around is common, which
   cuts against our pre-registered "No" (P(yes) 0.40). Must be cited.
2. Munirathinam, "Will the Agent Recuse, and Will It Stop?", arXiv 2606.06460 (v4 2026-08-01). In-band deny
   signal over SSH: recusal 100% to 55-75% by model; "an open-weights agent barely engaged"; a harness-level
   interceptor stops 120/120. Difference: SSH, not web; signal is a banner, not robots.txt; no canary.
   Closest analogue to "tool refuses, does the model route around".
3. Kim et al.? arXiv 2505.21733 (named in goal): log study, robots.txt compliance falls with stricter
   directives; AI-assistant fetchers often skip robots.txt. Fetcher-level, not model-level.
4. goodbot-badbot (github.com/dkd-dobberkau/goodbot-badbot): Disallowed honeypot paths, linked and unlinked,
   live crawler logs. Crawler side only. Disallowed trap directories are an old bot-trap technique.
5. "Identifying AI Web Scrapers Using Canary Tokens", arXiv 2605.13706: canary tokens for scraper attribution
   (title only checked so far).

Source facts (read in code, not measured):
- scrapy 2.13: linkextractors/lxmlhtml.py:160 records Link.nofollow; no CrawlSpider code filters on it.
- wget 1.25: src/recur.c:425 honours only META robots nofollow (GNU bug #59293); per-link rel ignored.
- mcp-server-fetch 2026.11.27: robots checked per model fetch (server.py:1097), refusal text server.py:834-840,
  loopback refused unless --allow-private-ips (server.py:184-210), Readability.js path when node on PATH.

Verdict so far: the crawler half is known by code; the new part is the model-level route-around rate after a
real tool refusal, with the defender's unconditional catch next to it. Not claimed by 2611.11306 or 2606.06460.
