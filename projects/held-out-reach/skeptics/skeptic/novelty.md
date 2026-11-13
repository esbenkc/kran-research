# Novelty check (Vera, skeptic) — defense-1113-held-out-reach-law

Claim under test (v2.1, felix/PREREG_v2.md): one per-agent option-use rate, measured on ONE fetch tool
(MCP fetch raw=true), times a model-free one-step reach crawl, predicts hidden-link follow rates within 10 points
on fetch tools the rate was never fit on.

Searched 2026-11-12 23:0x PST (web search; every arXiv id below opened on its abs page, not taken from a summary).

| Work | What it shows | Overlap with our claim |
|---|---|---|
| Chen, Siu, Liu, Song, Wang, "Tunable Tool-Call Rates in LLM Agents via Representation Steering", arXiv 2608.25198 (abs page checked) | Whether a model calls a tool is set by one residual-stream direction; the direction "generalizes to unseen tools". | Closest neighbour on "a per-model tool-use propensity transfers to unseen tools", but it is about calling vs not calling a tool on QA, measured inside open models, with steering. It does not measure behavioural option-use rates on shipped fetch tools, hidden links, or catch rates. |
| Khodayari, Zhang, Acharya, Pellegrino, "Indirect Prompt Injection in the Wild", arXiv 2604.27202 (abs page checked) | 70% of planted web instructions sit in non-rendered HTML; compliance across 13 models and 4 page representations depends on the representation. | Representation changes what models act on; no per-agent trait, no hidden-link follow rates, no cross-tool prediction. |
| Cloudflare, "Trapping misbehaving bots in an AI Labyrinth" (blog, Mar 2025) | Hidden links that "human visitors and legitimate browsers would never see". | The practitioner claim being audited; no measurement of agent follow rates per tool. |
| PalisadeResearch/llm-honeypot (GitHub) | SSH honeypot with ANSI-hidden prompts for LLM hacking agents. | Detection of LLM agents with machine-only content; different channel, no rate prediction. |
| Our own: invisible-link (defense-1108), hidden-link-crawl (defense-1109) | Stuck fit-link follow rate ~0.98-1.00 per agent; first-view crawl had 6 clear misses; raw-mode use 9/30, 24/30, 21/30. | This run reuses 1108's stuck rates and 1109's raw-mode cell as the calibration; the held-out tools and the two-factor rule are new. |

Verdict: NEW as far as this search reaches. No paper found that predicts agent follow rates of hidden (honeypot)
links across fetch tools from a per-model option-use constant. The nearest result on transfer of tool-use
propensity to unseen tools (2608.25198) is an internal-representation result on a different behaviour.
What would change this: a scraping/agent-security paper measuring per-tool honeypot-link catch rates for LLM agents
(none found in 2 searches; not exhaustive).
