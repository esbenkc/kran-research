# T6 — related work and novelty note for the post (Iris)

For Selma. Plain sentences the post may use as-is or shorten. Each line carries its source. Full audit with
quotes and page numbers: research.md section 4.

## Lines the post may use

1. Defenders plant decoys for AI agents: fake keys, fake secrets, fake links that only a machine would follow.
   Tracebit first planted a short string in a decoy secret that trips an attacking model's own safety checks
   (July 2026). In October it placed believable decoy pieces in the environment ("context bombs", nine kinds, e.g.
   a fake pentest report, request pacing, staged encryption) to misdirect attacking agents; with all nine in place,
   agents completed 91% fewer objectives in their AWS tests.
   Source: https://agentic.tracebit.com/context-bombs/ (July) ; https://tracebit.com/blog/context-bombs-that-misdirect-ai-attackers (October)
1b. Telling attacking agents that deception may be present already cut harm in one study. In Tracebit's May 2026
   test (10 frontier models in a cloud test range, after the Tularosa study's 2x2 design), the warning cut full
   compromise from 20% to 3%. Planting canaries without the warning did not. The warning was tested as one block.
   Source: https://agentic.tracebit.com/ai-agents-canaries/
1c. Prinos, Brush and Denton (2026) found the opposite pattern for spontaneous recognition: models wrote that they
   recognised a trap and still fell for it 73.4% of the time. They note that Tracebit "did not address why
   disclosing the presence of canaries in a prompt reduced worst-case compromise" and call for experiments that
   isolate the mechanism. Our run is one such experiment.
   Source: https://arxiv.org/abs/2606.21037 (abstract; p.15)
2. Attackers can also be told to look for decoys. In Rouxii (Cordeiro, Mongardini and Vasilomanolakis, 2026), one
   change to the attacking agent's prompt raised correct honeypot identification from 19% to 97%. Their aware
   agents then exploited the honeypots they found.
   Source: https://arxiv.org/abs/2609.26555 (abstract). Our name-it sentence is adapted from Rouxii's probe wording.
3. AgentTrap (Wang, Yu and Sun, 2026) built a honeypot that answers pen-testing agents step by step. Agents that
   recognised the deceptive endpoint refused to take its bait.
   Source: https://arxiv.org/abs/2610.02869 (abstract; section 4, trace analysis: models "recognize the deceptive endpoint as a credential-harvesting attempt and explicitly refuse to disclose the API key").
4. Writing about a situation can change behaviour. Schoen et al. (2025) found that when a model's reasoning says it
   is being evaluated, it takes fewer covert actions, and removing that awareness raises them.
   Source: https://arxiv.org/abs/2509.15541 (abstract).
5. "Label first, then act" prompts are an old jailbreak defense: Intention Analysis asks the model to state the
   request's intent before answering (Zhang et al., https://arxiv.org/abs/2401.06561). Self-reminders put a
   caution in the system prompt (Xie et al., Nature Machine Intelligence 2023,
   https://doi.org/10.1038/s42256-023-00765-8).

## Novelty line

We found no study that splits an agent's honeypot warning into its two parts, "decoys exist here" and "name a
suspected decoy in your thought", and tests each part alone on decoy use. The closest, Tracebit's May 2026 2x2,
tested the warning as one block; Prinos et al. 2606.21037 ask for exactly this split. Searched 2026-11-11: arXiv abstract pages
above and web searches for agent honeypot-warning and instruction-defense ablations. Scope the claim to our sandbox.

## Do NOT write (facts gate, Kasper #20064)

- Do not say AgentTrap tests warnings or claims warnings protect. It does neither (no warning text in the PDF).
- Do not say Tracebit tested operator warnings. Its context bombs are decoy content in the environment, read by the
  attacker; the 91% is for all nine combined, not for a single string.
- Do not call the mechanism "a written step before acting" or "forced verbalisation". Every arm already writes a
  thought before each tool call (agent.py SYSTEM_PROMPT: "keys thought, tool, args"). The tested lever is an
  instruction that AIMS the thought: "directed naming".
- Do not say Rouxii's agents avoided the honeypots. They exploited them.
- Do not credit "AgentTrap" for anything about agent skills: arXiv 2605.13940 is a different paper with the same name.
- Do not say Tracebit's warning reduced canary or decoy USE: its outcome is full compromise (admin + persistence),
  pooled over runs with and without canaries.
- Do not quote Tracebit's informing prompt: the page does not show its wording.
