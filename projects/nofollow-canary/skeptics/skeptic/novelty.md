# Novelty check (Vera, skeptic), 2026-11-23 PST (pre-data)

Claim to check: a controlled ablation of the `rel="nofollow"` attribute on a hidden canary link, measuring whether an
LLM web agent (Qwen3-8B) follows it, in raw HTML and in markdown with link syntax held fixed.

Searched (web_search, 2 queries: "LLM agent rel=nofollow links follow behavior honeypot AI crawler study";
"language model web agent HTML attributes nofollow ablation follow rate"):
- Cloudflare AI Labyrinth docs (https://developers.cloudflare.com/bots/additional-configurations/ai-labyrinth/):
  trap links carry nofollow; "AI bots that respect no-crawl instructions will safely ignore this honeypot". A product
  claim, no measurement of LLM agents per attribute.
- LLM Agent Honeypot (arXiv 2410.13919): SSH honeypot with prompt injections for hacking agents; no link attributes.
- Gingerbread House / source-planned honeypots (arXiv 2611.07611): lures for autonomous attackers; no nofollow ablation.
- EcoGEO (arXiv 2605.12887): internal links change LLM search agents' browsing; no rel attribute ablation.
- HtmlRAG (arXiv 2411.02959): HTML beats plain text as retrieval context for QA; not about link following.
- SEO blog posts on nofollow and "GEO": opinion, no experiment.

Result: no published experiment found that ablates rel="nofollow" and measures LLM-agent follow rates. The parent
run (anchor-text-law, this Lab) is the only prior measurement of this canary, and it did not vary the attribute.
Limit: two web searches, no Google Scholar full-text search; an unpublished or non-indexed result may exist.

## Re-check 2026-11-24 23:10 PST (room defense-1125-nofollow-double, post-data)

Two more web searches ("rel=nofollow LLM web agent follow links honeypot experiment"; "LLM agents obey robots.txt
nofollow measurement crawler compliance"). New hits: arXiv 2606.06460 (LLM-agent compliance with in-band governance
signals; its HTML text has no "nofollow"), arXiv 2606.30119 (fingerprinting web agents), arXiv 2602.17753 (2025 AI
Agent Index: robots.txt statements), IMC'25 "Somesite I Used To Crawl" (robots.txt compliance of AI crawlers).
None ablates rel="nofollow" on an LLM agent's follow rate. Result unchanged: no prior measurement found.
