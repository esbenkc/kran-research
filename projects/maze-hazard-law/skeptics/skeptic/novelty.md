# Novelty check (Vera, skeptic) — maze-hazard-law

Searched 2026-11-18 (web search, arXiv abstracts). Question: has anyone measured a per-page give-up rate (hazard) of
LLM agents in a decoy maze and tested whether a short pilot predicts the full run?

| work | what it measures | per-page hazard? | pilot -> full-run test? |
|---|---|---|---|
| Ord, arXiv 2505.05115 (2025) | constant per-minute failure hazard on METR tasks (agent success, not decoys) | yes (time, not pages) | no |
| Hamilton, "Peto's Paradox and the Future of AI Agents" (2026-01-23) | Weibull refit of METR data, shape k < 1 for all models | yes (time) | no |
| Ord, "Hazard rates for AI agents decline as a task goes on" (2026-02-04) | concedes a falling hazard | yes (time) | no |
| Pasquini et al., Mantis, arXiv 2410.20911 | tarpit for LLM attack agents, runs stopped at 10 rounds; "no attempts to backtrack or abandon" seen | no (capped) | no |
| Wang et al., AgentSnare, arXiv 2607.26998 (2026-07-29) | adaptive decoy env for pentest agents; 49.9% of tool calls absorbed, 57.2% of post-entry actions retained | no (aggregate shares) | no |
| Cloudflare AI Labyrinth blog (2025-03-19) | design + "four links deep" bot signal; no agent give-up number | no | no |
| Tracebit "Context Bombs" (2026) | aggregate cost multipliers | no | no |
| huntback.io ARTEX fork post (2026-11) | an open pentest agent with a built-in "tarpit breaker" (fetch budget 40 per host) | n/a | n/a |

Verdict: novel as framed (per-page give-up hazard in a decoy maze, censored survival fit, 3-page pilot vs KM median).
Must credit: AgentSnare (arXiv 2607.26998) is the closest decoy-cost measurement on LLM agents and is NOT in
PREREG section 7; Mantis for the "no backtrack" observation at 10 rounds.
Relevance note (not for this post's claim): agent-side tarpit breakers with a fixed fetch budget (ARTEX fork, 40)
make give-up a burst at a set depth by construction, the case the frozen rule detects only 9-18% of the time at n<=60
(power_weibull.txt).
