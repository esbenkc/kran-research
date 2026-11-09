# Novelty check (Vera, skeptic) — defense-1109-exposure-law-tools

Claim to check: a model-free crawl of shipped fetch tools (exposure = followable handle in the
first tool response) times a per-agent rate frozen from an earlier run predicts how often real
agent stacks follow a hidden link, per tool x hiding method x agent.

Searched 2026-11-08 (web_search, 3 queries + arXiv abstracts):

| Work | What it measures | Overlap with this claim |
|---|---|---|
| Khodayari, Zhang, Acharya, Pellegrino, "Indirect Prompt Injection in the Wild", arXiv 2604.27202 | 15.3K injections in 1.2B URLs; ~70% in non-rendered HTML; 5,200 runs, 13 models, 4 page representations; compliance up to 8%, structured representations reduce it | Closest prior: shows the page REPRESENTATION changes model compliance. Instructions, not links; no model-free predictor of the rate. Relevant to the 'No' side (framing matters). Must be credited. |
| Fayolle et al., "On the Internet, Nobody Knows You're an LLM Bot", arXiv 2606.30119 | 6 web agents on honeysites; network/TLS/browser fingerprinting separates agents (F1 0.931 browser layer) | Treats the stack as the unit; detection by fingerprint, not by hidden-link follow. No overlap in measure. |
| "Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces", arXiv 2605.14786 | behavioural fingerprinting of browser agents | Different signal (UI traces). |
| WebCloak (web-cloak.github.io) | obfuscation + 'semantic labyrinth' against LLM scrapers | Defense via misdirection; no per-tool exposure x follow law. |
| Cloudflare AI Labyrinth docs; ScrapingBee / Browse AI / Bright Data honeypot guides | assume bots see hidden links / say rendering agents skip them | Untested practitioner claims: these ARE the two published baselines (all-1, all-0). |
| Parent: kran.ai invisible-link (defense-1108) | follow rates with hand-made raw/md/a11y tools | Same carrier; this run moves it to shipped tools and adds the model-free predictor. |

Verdict (pre-data): no prior work found that predicts per-stack hidden-link follow rates from a
model-free crawl, or that measures hidden-link exposure across shipped converters (mcp-server-fetch,
Readability, turndown, crawl4ai, browser-use DomService, Playwright). The representation-effect
finding of arXiv 2604.27202 is prior evidence that framing can move rates and must be cited.
Limit: 3 searches; not a systematic review.
