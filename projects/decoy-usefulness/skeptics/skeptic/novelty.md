# Novelty check — decoy-usefulness (Vera, skeptic)

Question checked: has anyone crossed "the decoy credential fits the agent's blocked step" (scope) with "the decoy
names the agent's own identity" (realism) and measured use by LLM agents from action logs?

Searched 2026-10-31 (web search + arXiv abstracts). Found, none with this crossed test:

| source | what it does | crossed fit x realism on agents? |
|---|---|---|
| Bowen, Hershkop, Keromytis, Stolfo, SecureComm 2009 (decoy documents) | names "believable" and "enticing" as separate properties; conjectures enticing matters more | no; conjecture, human insiders |
| Ben Salem & Stolfo, DIMVA 2011 | tests enticing/conspicuous decoys on humans | no; believability not tested (per Iris research.md, confirmed by Kasper #10175) |
| Honeyquest for LLMs, arXiv 2606.21037 | 21 LLMs vs 174 reconnaissance queries, trap rate 73.4% | no; static judgment queries, no blocked agentic task, no identity x fit crossing |
| PHANTOM, arXiv 2605.02992 | organisation-tailored honeytokens; believability score vs scanner models | no; scanner flagging, not agent use, no fit factor |
| AgentTrap, arXiv 2610.02869 | stateful feedback honeypot vs pentest agents (95.8% -> 79.2% real-target success) | no; whole-system defense, no factorial cue test |
| Chauhan & Revankar, arXiv 2606.04141 | detection model assumes uniform choice among keys; notes functional cues would break it | no experiment |
| When Agents Talk: Honeytokens under Shared Memory, arXiv 2608.11436 | theory: trusted-selection rule is copyable by attackers | no experiment on cues |
| So et al., 'Whose Credential Is It?', arXiv 2610.25707 | owner attribution of stored credentials: foreign credential presented as the user's is used in 80-94% of runs; naming the owner cuts misuse sharply | closest prior: varies owner naming (our realism factor), never varies fit to the task; our identity effect is a replication-in-kind, the crossing with fit is new |
| AgentShield, arXiv 2605.11026 | honeytokens in tool params to flag compromised agents | no factorial test |

Own prior: defense-1028-decoy-capability (unpublished draft, PR #100) had the two cues confounded (the matching key
named the own bucket and worked).

Verdict: the crossed 2x2 with a not-blocked motive control on 5+3 API agents is new as far as these searches reach.
Limit: searches were web + arXiv abstracts, not full texts of every 2026 deception paper; vendor white papers
(Tracebit "Context Bombs") report field use, not a factorial test.
