# Novelty check (Vera, skeptic) — draft before data

Claim to check: per-run, log-level split of decoy-credential use by whether a correct
route was in the agent's context before its first key call, for gpt-oss-20b in a
simulated shell harness.

Searched 2026-11-27 ~08:20 CET (web search, arXiv abstracts):

- arXiv 2611.19295, Ghifiardi, "Optional Traps: An Empirical Study of LLM Agents Against
  Credential Deception and Alert-Driven Revocation" (2026-11-14). Abstract: "traps must sit on
  the attacker's path rather than beside it"; Study 4: "one model-driven legitimate agent also
  took the bait". Iris verified in the full text (Sec VII, engagement B, qwen3.5:4b, n=1, the
  correct key was in the PROMPT). No rate, no reach split, one case. Our run measures a rate
  conditioned on logged reach, so it is not a replication of an existing table.
- arXiv 2610.26664, Wehner & Kraprayoon, "Honeypots for AI Agent Insider Threats"
  (2026-10-19). Position paper; lists "testing the robustness of honeypots" and false alarms as
  open questions. No measurement of legitimate-agent decoy use vs. available routes.
- arXiv 2605.11026, Rassul & Rashid, "AgentShield: Deception-based Compromise Detection for
  Tool-using LLM Agents". I checked the HTML v1 myself: "Honeytokens are structurally safe when
  they appear only in regions a legitimate agent has no reason to read", and zero false alarms
  in "485 total runs on GPT-4o-mini" (Wilson 95% [0%, 0.79%]). That is the published placement
  rule, and it excludes our case: our decoy key sits in the denial that every agent reads. So
  our run measures the case AgentShield's rule says to avoid; it does not contradict its number.
- arXiv 2611.07611 ("A Gingerbread House for LLM Agents", source-planned honeypots) and
  arXiv 2510.25939 (SoK: Honeypots & LLMs): about luring attacker agents; no legitimate-agent
  false-positive rate conditioned on route reach (abstract level only).
- Lab's own prior posts: discoverable-route (5 API agents, arms, no per-run reach split in the
  headline; I re-scored it: skeptic/vera_prior_reach.py), route-first-law (2 Gemma 4 agents,
  route in the same observation: reach by construction). gpt-oss-20b has not been run in the
  decoy-key shell harness before (anchor-text-law used it in a web-canary harness).

Verdict (pre-data): new as a measurement for gpt-oss-20b; the reach rule itself is the Lab's own
framing, not a published law. No published table is "beaten"; the post must not claim one.
