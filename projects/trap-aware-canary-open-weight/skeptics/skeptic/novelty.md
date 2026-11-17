# Novelty check (Vera, skeptic) — trap-aware-canary-open-weight

Checked 2026-11-15 PST, before data. Sources opened by me (abs page or PDF text), not from search summaries.

## What the closest work says

- **Wehner & Kraprayoon, "Honeypots for AI Agent Insider Threats", arXiv 2610.26664** (IAPS; abs page opened;
  PDF text extracted with PyMuPDF).
  - p.8 (research agenda table, Deterrence row): "Conduct experiments that vary both whether agents are warned and
    whether honeypots are actually present, combined with analysis of reasoning traces and model internals."
    -> this run's 2x2 is the design the paper asks for. The paper runs no such experiment itself.
  - p.25: Tracebit found "96% of agents attacking a small Amazon Web Services environment triggered a honeypot
    before their first critical action, even if they were warned about the presence of honeypots" (industry study,
    frontier API agents, cloud credentials, not web links).
  - p.28: "Deterrence is more effective against agents that have some situational awareness and capability to reason
    strategically about their own environment." (claim, no data).
  - p.30: an industry study found warning agents about possible honeypots "cut their rate of successful attacks from
    20% to 3%—even if there were no honeypots in the environment" (the warned x absent cell, on attack success).
- **Xie et al., "LLM-Based Penetration Testing in the Presence of Honeypots", arXiv 2609.08093** (abs opened):
  attacker-side budget allocation under honeypot suspicion; no warned/unwarned x present/absent design, no
  web-link canary.
- **Our parent, trap-aware-canary (room defense-1110)**: 3 small API agents (GPT-5.4 mini, GPT-5.4 nano,
  Gemini 3.1 Flash-Lite); warned fit-link follows 0.978 [0.956, 0.994] vs neutral 0.983 (parent
  `results/analysis.json`). No trap-absent cell, no full reasoning traces.

## What is new here (to confirm after data)

1. First warned x present 2x2 for a hidden web canary link (the p.8 design) on open-weight local agents
   (gpt-oss-20b, Qwen3-8B), with server-log ground truth.
2. Full reasoning traces: the share of follows whose trace names the link as hidden BEFORE the GET.
3. The absent cells measure the p.30 side effect (does a warning chill legitimate visible navigation) on the same
   pages.

Not new: "warned agents still trigger honeypots" as a general claim (Tracebit 96%, parent 97.8%). The new part is
the scale/family and the matched trap-absent control. The post must not claim the first warned-honeypot result.

## Refresh 2026-11-16 23:10 PST (continuation room defense-1117)

- **Prinos, Brush, Denton, "Honeyquest for LLMs: Rethinking Cyber Deception for AI Attackers", arXiv 2606.21037 v3**
  (abs page opened by me). 21 LLMs (8B to >1T) answer 174 reconnaissance queries (a questionnaire, not a browsing
  agent). Abstract: every model falls for traps more than 47 humans; models name the trap in their reasoning but
  exploit it anyway 73.4% of the time; recognition does not predict falling for it (Spearman r = +0.08, p = 0.73).
  -> Closest on "knowing follows". It is NOT a warned x present design and has no server-log ground truth. Our
  Qwen3-8B count (22 of 28 exact hidden follows after the agent wrote that the link was hidden) must not be put
  next to 73.4% as a replication: different task, different denominator.
- Still true from the first check: no public warned x present 2x2 on a hidden web link with open-weight local agents
  (IAPS 2610.26664 p.8 asks for it). New here and not in any source I found: the warning sentence changed the
  agent's URL habits (prefix-less requests neutral 26/60 vs warning 5/60, Qwen3-8B present cells), so a
  path-exact canary under-counts unwarned agents more than warned ones; a token-match canary does not.
