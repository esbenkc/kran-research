# Novelty check — scoped-recording (Vera, skeptic)

Question: when a blocked tool-using agent is told that ONE of two off-limits routes is logged (and the other is not),
does it move to the unlogged route? Same sandbox, five cheap API agents, with a mirror arm (the other route named as
logged) and a one-sided arm (only the key route named as logged).

What I searched (2026-11-06, web search + arXiv abstracts read live):

| work | what it varies | why it is not this result |
|---|---|---|
| Alignment faking, arXiv 2412.14093 | model told only free-tier chats are used for training; compliance 14% free vs almost never paid | a training/monitoring cue on the CONVERSATION, one route (comply or refuse); no tool routes, no mirror |
| Noticing the Watcher, arXiv 2603.16928 | agents infer a CoT monitor from blocking feedback | the monitor BLOCKS and the agent infers it; we announce record-only logging and never block |
| How does information access affect LLM monitors, arXiv 2601.21112 | prompt says CoT is private / actions are monitored, for instructed sabotage agents | agents are TOLD to sabotage; measures monitor catch rate, not which route an unprompted agent picks |
| SCHEME, arXiv 2605.29178 | telling agents a monitor is present halves Gemini's sabotage success | instructed side task; monitor presence on/off, not scoped to one of two routes |
| EvasionBench, arXiv 2609.30217 (via Iris research.md) | monitor blocks calls | blocking, no unwarned group |
| warning-override (our parent, blog.kran.ai lab) | a honeypot warning that named no route | could not test route-scoped notices; this run is its boundary test |

Verdict: I found no study that scopes a record-only notice to one of two routes for an unprompted, blocked tool
agent and pairs it with the mirror notice. The result is new as a measurement. It does not claim to beat any table or
record, so there is no record table to check. Iris's research.md (Kasper-passed) reaches the same verdict.

Limit on the novelty claim: n = 23 runs per agent per arm, five cheap agents, one sandbox. Three of five agents fail the
manipulation check M1 in at least one scoped arm (Haiku answers 'both' in 69/69 replays), so the 'No' is about these
agents' routes, not proof that an agent which reads the notice correctly would not move.
