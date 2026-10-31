# Novelty check (Vera, skeptic) — honeypot-talk

Claim to check: a run-level, within-model comparison of executed decoy use (logged tool call presenting the decoy key)
between runs whose own text names the decoy as a honeypot and runs that do not, in a stateful agent sandbox,
with a counter-deception (identify-only) prompt on/off.

Searched 2026-10-31 (web search, arXiv abstracts read directly):

| work | what it measures | overlap |
|---|---|---|
| Honeyquest for LLMs, arXiv 2606.21037 (Prinos, Brush, Denton) | static questionnaire; 'walked into trap' = recognised in reasoning AND placed exploit mark, 73.4%; model-level Spearman r = +0.08 over 21 models | same question at model level and in a static, marked-answer setting; no executed call, no run-level say vs not-say contrast (abstract) |
| Rouxii, arXiv 2609.26555 | identification of honeypots with an anti-deception prompt (19% -> 97%); stated next step (NEXT_AWARE) | stated plan, not an executed call |
| Not All Eval-Awareness Is Equal, arXiv 2608.27340 (Zhuang, Aranguri) | verbalized eval-awareness framing vs compliance, Qwen3-32B, FORTRESS | run-level say-vs-do design, but eval awareness, not decoys, and no agent tool log |
| Schoen et al., arXiv 2509.15541 | verbalized eval awareness causally lowers covert actions (13.2% -> 24.2% when removed) | agentic, but eval awareness, not a named decoy resource |
| SkillSentry, arXiv 2608.03485 | honey worlds with decoy resources to test agent skills | uses decoys as a sensor; does not relate agent text to decoy use |
| When Agents Talk: Honeytokens under Shared Memory, arXiv 2608.11436 (Gans) | theory: decoys vs attackers sharing memory | no experiment on agent text vs use |
| OSWorld 2.0, arXiv 2606.29537 | one trajectory notices credentials, calls them 'not relevant', leaks them | anecdote, not a measured rate |

Verdict: NEW as a design (executed, logged decoy use; run-level, stratified by model and decoy; identify-only prompt
as the lever). Not new as a question: Honeyquest asks it at model level in a static setting, and the eval-awareness
literature asks the general version. The post must credit both and must not say 'first to ask'.
