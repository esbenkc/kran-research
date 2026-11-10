# Novelty check — Vera (skeptic), defense-1110-trap-aware-canary

Searched 2026-11-09 ~23:00 PST. Question: does a hidden, task-fit canary link keep catching agents that a
system-prompt sentence warns about trap links?

## Closest prior work found

| Work | What it tests | Overlap | What it does NOT test |
|---|---|---|---|
| Rouxii, arXiv 2609.26555 (2026-09-22) | Prompt-only "counter-deception" pentester vs vanilla; honeypot identification 19% -> 97%, false alarms 0.7% | A prompt that makes the agent aware of traps changes its behaviour (identification) | Network honeypots, not web links; identification rate, not whether the agent still takes a trap that is the only route; no watcher (logging) control |
| Tracebit, "How Tracebit's Context Bombs derail offensive AI" (2026-10-21, tracebit.com/blog/context-bombs-that-misdirect-ai-attackers) | "Pentest report exposure" bomb labels real targets as canaries; GPT-5.6 Cyber and Kimi K3 reach 0 objectives, Claude Opus 5 -11%, GPT-6 Astra unchanged; ~10 runs per model per bomb | Trap talk in the environment causes avoidance, and the size differs a lot between models | Warning sits in the environment, not the agent's system prompt; targets are not a sole-route task-fit link; no watcher control |
| Palisade LLM Agent Honeypot, arXiv 2410.13919 | Prompt-injection traps detect AI agents in the wild | Canary-style detection of agents | Never warns the agent about traps |
| Cloudflare AI Labyrinth (blog.cloudflare.com/ai-labyrinth, Mar 2025) | Hidden links for suspected bots, follow = bot signal | Hidden-link detector | No published test against agents told about traps; markup not published |

## Verdict

The DIRECTION "trap awareness -> avoidance" has prior support (Rouxii, Tracebit pentest bomb). The new bits:
(1) the trap is the ONLY route to the task answer and fits the task, so avoidance has a cost;
(2) a matched watcher sentence (logging) in the same system-prompt slot isolates the "trap" label;
(3) server-log ground truth per run. A post that claims "first to show warnings change agent behaviour" would
be wrong; it must credit Rouxii and the Tracebit pentest report.
