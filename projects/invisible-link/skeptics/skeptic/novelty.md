# Novelty check (Vera, skeptic) — invisible-link

Question: do LLM browsing agents follow a CSS-hidden (display:none + aria-hidden) footer link more often when its
anchor text fits their task, by agent view (raw HTML, markdown, accessibility tree)?

Searched 2026-11-07 (web_search, 4 queries: hidden-link honeypot LLM agents; accessibility tree hidden link agent
detection; Labyrinth follow rates; arXiv lookups). Independent of Iris's research.md.

| source | what it measures | overlap with ours |
|---|---|---|
| Cloudflare AI Labyrinth (blog.cloudflare.com/ai-labyrinth, 2025-03; developers.cloudflare.com/bots/additional-configurations/ai-labyrinth) | design: invisible nofollow links to generated pages for non-compliant crawlers | no follow rates published; no LLM-agent split by view or anchor fit |
| Nitu, Mühle, Stöckl, arXiv 2507.12844 (Machine-Readable Ads) | Browser Use agents click ad banners when off-screen text labels exist | hidden-but-in-a11y labels reach agents (our sr-only control); not hidden links, not task fit, not raw/md views |
| arXiv 2508.01390 (LLM pollution of online behavioural research) | recommends hidden honeypot fields for survey agents | recommendation, no link-follow rate by fit |
| arXiv 2409.11295 (EIA) | hidden-element prompt injection on web agents | attack with instructions; ours is an instruction-free link (defensive measurement) |
| arXiv 2609.16732 (UI desynchronization, mobile agents) | tiny/transparent honeypot widgets in the a11y hierarchy, with misleading text | mobile a11y injection; not web links, not fit x view |
| arXiv 2510.18113 (dark patterns on LLM web agents) | agents fall for deceptive UI | not hidden links |
| Practitioner scraping guides (scrapingbee, brightdata, scrapfly, 2025-26) | claim: parsers reading raw HTML follow hidden links; rendered checks (is_visible) skip them | claims only, no LLM-agent measurement |

Verdict: I found no published measurement of hidden-link follow rates for LLM agents split by anchor task fit and by
the agent's page view. The novel parts are (1) the fit gap on an instruction-free hidden link, (2) the raw-HTML
cell, where the hiding cue is in the text the model reads, and (3) the route-start need contrast. The a11y
display:none zero and the crawler 100% are known by construction (spec / code) and are controls, not findings.
Gap in this check: I did not search vendor bot-management reports (Akamai, HUMAN, DataDome) beyond the top results.
