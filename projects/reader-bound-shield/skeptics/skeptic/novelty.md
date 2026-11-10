# Novelty check (Vera) — reader-bound-shield

Claim to check: on one RL engine, the probe-off switch that label-free reward teaches is at most the edge of a
labelled reader of the same size on the same trace (y <= x), tested where the text difference is ~10 points but the
reader edge differs (planted one-token cue x ~ f vs real probe x ~ 0).

Searched 2026-11-09 (web search + arxiv.org abstract pages; export.arxiv.org rate-limited):

| paper | what it does | overlaps? |
|---|---|---|
| Baker et al. 2503.11926 (Monitoring reasoning models for misbehavior) | CoT monitor in RL reward -> obfuscated reward hacking with more optimisation | threat model only; no probe-sensing switch, no reader bound |
| RL-Obfuscation 2506.14261 | RL against latent-space probes; some probe types evaded | probe-in-reward evasion of the probe's READING; not sensing whether the probe is attached |
| Neural Chameleons 2512.11949 | fine-tune to evade unseen probes given an explicit trigger phrase "You are being probed for X" | the trigger is given in the prompt; here the model must infer the probe from a numerical serving trace |
| You Can't Escape Your Activations 2609.03035 | agents TOLD that an activation monitor is present; detection with fixed probes | awareness is given by instruction, not learned from a trace |
| Alignment via Training Against Probes Without Losing Monitorability 2609.38645 | probe-guided fine-tuning, monitorability kept | probe as training signal; no switch, no reader |
| Designing Effective Monitor-Based Interventions 2611.09828 | monitors in RL against reward hacking (Qwen3-4B) | monitor design; no serving-trace channel |
| Obfuscation Atlas 2602.15515 | where honesty vs obfuscation emerges in RLVR with deception probes | evasion of the probe reading; no probe-attached detection |
| Shalev-Shwartz et al. 1703.07950 (Failures of gradient-based deep learning) | gradient learners fail on parity-like targets | theory backdrop for a hash cue x ~ 0 and y ~ 0; not an RL-vs-labelled-reader comparison |

Our own earlier Lab posts on the same engine (multi-token-recompute, long-rollout-shield) measured the reader edge
(real probe ~0 vs one-token cue 10.7) but had no working positive control. So far I found no published test of
"reward-taught switch <= same-size labelled reader edge". Verdict on novelty: NEW as a measured bound on this engine;
the theory side (the policy is a member of the reader class, so y <= x is about optimisation, not capacity) is a
one-line argument, not new.
