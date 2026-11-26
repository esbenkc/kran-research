# Novelty check (Vera, skeptic) — draft before data

Claim to check: a randomized test of WHERE a correct route sits relative to a decoy inside the one
observation where the decoy first appears (above vs below; same bytes otherwise), in a web-canary
harness and a decoy-key shell harness, with open-weight Gemma 4 agents.

Prior work found (search 2026-11-26, web_search; links opened via search snippets):
- Optional Traps, arXiv 2611.19295 (2026-11-14): Study 1, a 9.7B agent touched 0 decoys because a real
  leak answered first (n=1 per condition). Study 2 removed the leak; a 4B agent touched the planted
  canary in 19/20 defended sessions. Conclusion: "traps must sit on the attacker's path". This is a
  PRESENCE contrast (leak present vs absent), not a position contrast, and n=1 in the leak cell.
  https://arxiv.org/abs/2611.19295
- Cloak, Honey, Trap (Ayzenshteyn et al., USENIX Security 2025): LLM agents follow blatant traps.
  No route-position manipulation.
  https://www.usenix.org/system/files/usenixsecurity25-ayzenshteyn.pdf
- A Gingerbread House for LLM Agents, arXiv 2611.07611: source-planned honeypots that lure attackers.
  Lure design, not order vs a real route.
- "Does Rank Still Matter? Position Bias When AI Agents Shop on Our Behalf": item-position effects on
  agent choice vary by model. Product choice, not decoy vs real route, no security harness.
- The Hidden Dangers of Browsing AI Agents, arXiv 2505.13076: start/end attention bias in browsing
  agents (injection placement). Attack placement, not defender decoy placement.
- Our own: discoverable-route (presence: key use 96% -> 17-18% with the route in the runbook),
  anchor-text-law and nofollow-canary (observational menu-first split, Qwen3-8B 3/19 vs 69/71 in raw
  HTML, my rebuild vera_parent_order.py).

Verdict (pre-data): the position contrast with the same route above vs below the decoy, randomized,
in two harnesses, is not in the sources above. To finalize after data: re-run the search with the
post's final claim words.

## Final (after data, 2026-11-26)

Second search (web_search, "decoy placement relative to legitimate route LLM agent canary position") found, besides
2611.19295: AgentShield (arXiv 2605.11026, deception-based compromise detection for tool-using agents) and Agentic
Cloud Decoys (arXiv 2607.24006). Neither moves a decoy above vs below a live route inside one observation. Iris's
research.md F6 lists BiasBusters (arXiv 2510.00307: earlier-listed tools favored) and Lost in the Middle
(arXiv 2307.03172) as position-bias priors; our 26B web and 31B shell cells go the other way (the later item won).

Verdict: new. The finding (a correct route below the decoy kept the decoy at 10% web and 0% shell, so the
"route first" placement rule is not needed; one model per harness took the trap more when it came last) is not in
the sources above. It directly tests and narrows the 2611.19295 claim "a trap only works if it stands where the agent
must go": a live route anywhere in the same page or denial beat the trap here. Claim limits: one family, our sandbox.
