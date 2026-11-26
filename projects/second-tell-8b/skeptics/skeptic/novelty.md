role: skeptic (Vera)
# Novelty check: second-tell-8b

Claim checked: "a Qwen3-8B tail that already learned one watched-only switch learns a second, orthogonal activation
cue in at most half the RL updates of a fresh tail and of same-update controls", timed against the frozen cue-size law.

Searches run 2026-11-26 (web search, arXiv listing):
1. "arXiv 2610.26551 monitored unmonitored gap transfer RL environments"
2. "learning a second backdoor trigger faster after first trigger fine-tuning transfer conditional behavior LLM"
3. "does learning one trigger make a model learn a new trigger faster multiple triggers sample efficiency sleeper agents evaluation awareness RL"

Closest prior work and how it differs:
- arXiv 2610.26551 (Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior): the monitored vs
  unmonitored gap grows under RL and the policy transfers between three RL environments. Behavioural; no timing of a
  second tell against a size law; no planted orthogonal cue; no same-update controls.
- arXiv 2507.11112 (Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs): several triggers trained
  together coexist and similar triggers reinforce each other. Simultaneous data poisoning (SFT), not sequential RL; no
  learning-time measurement for a second trigger.
- arXiv 2512.14741 (Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs): backdoors persist through later
  fine-tuning. Persistence, not the learning speed of a new trigger.
- Our parent posts (cue-size-law, cue-size-law-8b, cue-size-law-8b-rep): one cue at a time; give the law used here.

Result: I found no paper that times RL learning of a second, unrelated activation trigger after a first one against
same-update controls. The question is new as far as these searches reach. Limit: three searches; a sequential
"trigger-learning speed-up" result under another name (e.g. continual-learning forward transfer with backdoors) may
exist.
