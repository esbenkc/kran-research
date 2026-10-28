# Novelty check (Vera, skeptic) — draft, 2026-10-28

Question: does the static-vs-stateful decoy gap change with agent capability (an interaction on a capability axis)?

Searched (2026-10-28 ~00:10 PDT): arXiv abstracts + web search "LLM agents honeytoken decoy trip rate capability scaling stateful deception".

| Work | What it measures | Capability axis? | Static vs stateful? |
|---|---|---|---|
| Honeyquest for LLMs, arXiv 2606.21037 (Prinos, Brush, Denton) | 21 LLMs, 8B to 1T+, 174 reconnaissance queries (judgment questionnaire, 10,962 responses) vs 47 humans; every model falls for traps more than humans; recognition-action gap 73.4%; recognition does not predict falling (Spearman +0.08) | 21 models, but the abstract states no capability trend as a measured slope | No (static questionnaire items) |
| AgentTrap, arXiv 2610.02869 (Wang, Yu, Sun) | 8 pentest agents vs one web app; stateful deception cut real-target success 95.8% -> 79.2%, elicited attacker API keys in 18.8% of runs, beats static deception and fixed escalation | No slope reported (abstract) | Yes |
| AgentLSD, arXiv 2609.19140 (Golinelli et al.) | 6 models, 11 web CTFs, injected fake flags / decoy endpoints; traps add +20 turns, +2k reasoning tokens; solve-rate effects heterogeneous | 6 models, no slope in abstract | No |
| LLM-Based Pentesting in the Presence of Honeypots, arXiv 2609.08093 | attacker-side honeypot-aware budget allocation | No | No |
| Cloak, Honey, Trap (CHeaT), USENIX Security 2025 | traps/honeytokens vs LLM agents on 11 CTF machines | No | No |
| Tracebit "Context bombs" (agentic.tracebit.com/factory) | fake pentest report stopped some models, not GPT-6 Astra; spend per objective up to 23x | Anecdotal model contrast | Static lure relabels real targets (different mechanism) |

Preliminary verdict: no work found that reports a static/stateful x capability interaction for decoy trips. Novel IF the post claims only the measured slope on its own ladder. To re-check at sign-off against what the post claims to beat.

## Verdict at sign-off (data @263af97)

The result the post can claim: on 8 API agents, the gap between using a state-matching key and a fixed-content key
grows with task capability (I = -0.66, 90% t [-0.98, -0.35]); the static key stays near the floor at every rung.
None of the works above reports how the static/stateful gap changes with capability. AgentTrap (2610.02869) reports
stateful > static but no capability axis; Honeyquest (2606.21037) reports no capability trend for static items.
So the measured interaction is new as a measurement on this ladder. It does NOT beat or contradict a record table:
no prior number exists for the same quantity. The post should say "first measurement on this ladder", not "beats".
The title's premise (static use falls with capability) is not supported, so the post cannot claim support for the
Tracebit "strong models ignore static lures" observation either.
