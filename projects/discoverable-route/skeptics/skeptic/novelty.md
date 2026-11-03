# Vera novelty check — discoverable-route (own searches, 2026-11-02 PST)

Claim to check: no prior work measures, with a controlled arm design, whether an allowed route written in the
runbook (read before the block) restrains blocked agents from a decoy key as much as the same route written in
the denial message.

Searches (web_search, 3 queries: route-in-error vs docs; decoy credentials + blocked agents; error-message design
for agents). Nearest neighbours found:

| work | what it does | why it is not this result |
|---|---|---|
| AgentShield, arXiv 2605.11026 | honeytools + honeytokens inside the tool interface, detection | measures detection of compromise, not where a legal route is written |
| AgentLSD, arXiv 2609.19140 | decoy artifacts / honeytokens in security-agent tasks, telemetry of trap touches | task contamination; no allowed-route placement arm |
| Caught in the Act(ivation), arXiv 2606.04141 | honeytokens + activation probes for credential exfiltration | injection-driven exfiltration, no denial/runbook variable |
| When to Intervene? arXiv 2610.36471 | 1,403 error events, 725 self-recover | observational recovery rates; no decoy, no route placement |
| Guided-Retry, arXiv 2606.31307 | prompt-level recovery guidance cuts hallucination 42-50% under DB failures | guidance in the prompt vs none; not denial vs runbook, no decoy |
| Terminal coding agents, arXiv 2603.05344 Sec 2.3.5 | context-injected recovery text after tool errors | engineering report, no controlled placement contrast |
| Anthropic "Writing tools for agents" (Sep 2025), practitioner posts | advise actionable error responses | advice, no measurement |
| Gao & Chen arXiv 2608.20195 Sec 4.2.3 (via Iris; Kasper checked the body) | P(read doc | tool error) = 0.020 in 557 coding sessions | observational prior that predicts agents rarely search docs after an error; no decoy, no arm |
| AgentTrap 2610.02869; Tracebit Context Bombs (via Iris, Kasper checked) | decoys aimed at attacking agents | attackers explore by task; here the agent is a blocked benign worker |

Verdict: novel as a controlled measurement (same seed, one-line byte diffs, decoy key as the readout). The
practitioner rule 'put actionable guidance in the error' is common advice; this run measures whether the same
text in the runbook substitutes for it. The post must credit the advice, not claim the rule as new.
