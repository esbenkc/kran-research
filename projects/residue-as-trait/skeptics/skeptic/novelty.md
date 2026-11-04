# Novelty check (Vera, 2026-11-04)

Claim checked: "an agent's leftover decoy-key use carries over to a new task" (D_add held-out test,
5 agents x 8 tasks x 2 route placements, after a route fix).

Closest prior work found (abstract pages opened):
- Krakovna, Lindner, Ho, Farquhar, Shah, "Realistic honeypot evaluations for scheming propensity",
  arXiv 2605.29729 (2026-05-28). Honeypot coding tasks in internal codebases; reports that scheming
  propensity is higher for Gemini 3.1 Pro than 3.0 Pro and that this is consistent across their honeypot
  evaluations (search snippet of the PDF, Figure 3). That is a model-level ranking observation across
  honeypots for two models, under agency prompts. NOT in research.md -> must be credited.
- Wang et al., arXiv 2607.28685; Zhang et al., arXiv 2605.23950 (in research.md): rankings flip across
  benchmarks / harness variance can exceed model variance. Our result is consistent with a narrower
  reading: the agent offset carries across tasks within one decoy design, but the route placement
  reshuffles agents (cross-arm runbook->named D -0.071; nano 12/24 named vs 4/24 runbook).
- "Are LLM Agents Behaviorally Coherent?" arXiv 2509.03736: agents fail to sustain trait-driven
  coherence across settings (personality traits, not decoys).

Verdict: the held-out, shuffle-null test of cross-task carry-over for a post-fix decoy is new as far as
I found. The post must NOT say "first disposition measure from decoys" (the goal's 'if it holds' line):
2605.29729 already reads honeypot rates as a per-model propensity. Say "a held-out test" and cite it.
