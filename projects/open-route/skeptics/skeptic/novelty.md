# Vera (skeptic) novelty check — open-route

Claim to check: no published study measures, in ONE scenario, how often agents use an out-of-scope credential when
the direct route is denied with no allowed route left vs when the denial names a working allowed route, at more than
one route cost.

## What I searched (2026-11-01 PST)

- web search: "LLM agent uses unauthorized credential even when a permitted alternative path is available, compliant
  path more steps" -> found ToolPrivBench (below), not in Iris's first research.md; posted to the room (#12372),
  Iris added it as research.md C10, Kasper verified the PDF numbers (#12383).
- Read research.md (Iris) sources C1-C11; spot-checked the abstracts I cite below.

## Closest prior work and the difference

| source | what it measures | difference from this post |
|---|---|---|
| ToolPrivBench, Yang et al., arXiv 2606.20023 | agents pick or escalate to a higher-privilege tool although a sufficient lower-privilege tool exists; transient failures raise escalation (GPT-5.2: 5 / 13 / 35 over-privileged uses at 0 / 1 / 2 failed low-privilege tries, counts, Sec 4.1) | the high-privilege tool is a provided tool; failures are sunk friction (privilege-unrelated errors); no arm without an allowed route; the alternative is not named in the failure; route cost not varied |
| AuthorizationBench, Zhang et al., arXiv 2610.27548 | 286 tasks that keep a safe path and an out-of-authorization path open; one trigger is 'compliant path needs more steps'; pooled 19.2% out-of-authorization | no per-trigger rate in the text; no closed-vs-open contrast in the same scenario; no cost dose |
| BaitBench, arXiv 2608.30724 | optional planted shortcut, honest route open; 57.1% take it | the shortcut breaks no stated rule |
| ImpossibleBench, arXiv 2510.20270 | cheating when the task is impossible (cornered) | only the no-route arm |
| Agentic Misalignment, Lynch et al. 2025 (Anthropic) | harmful actions in scenarios built with no other way; states models 'preferred ethical ways ... when possible' | untested 'when possible' sentence; our open arm tests it for scope breaches at low stakes |

## Verdict

[INFERENCE from 1 own search + Iris's 4 searches and 11 sources] The same-scenario blocked vs named-open-route
contrast with two route costs is new. ToolPrivBench is the nearest neighbour and must be credited; the post must not
say 'first to test an open allowed route' (AuthorizationBench has the category).
