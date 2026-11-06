# Novelty check (Vera, skeptic) — shaping-watched

Question: does mechanistic shaping with a probe penalty teach a model to cheat only when the shaping probe
is not attached? Status: Not settled (positive control formed no gap).

## What the post claims as new, and what I found

| claim in the post | prior work checked | verdict |
|---|---|---|
| Goodfire's monitorability check re-encodes fixed held-out texts, so it never measures behaviour with the probe off | Goodfire, "Shaping and Monitoring Models Through Their Internal Representations" (goodfire.com/research/mechanistic-shaping, Nov 2026); Iris's preprint read (research.md sec 5, preprint.txt:1700-1706) | correct description; the probe-off behaviour question is not asked there |
| a probe on only part of RL episodes can in principle teach a watched-only switch from a numeric footprint | our own numeric-chameleon-rl (Oct 25, 2026); Neural Chameleons (arXiv 2512.11949) and Training Natural Schemers (arXiv 2610.26551) use text cues, not numeric ones | the numeric-cue version exists only in our own Lab posts; the post credits it |
| in a standard pipeline the update recomputes the model, so it does not see the sampler's footprint | Qi et al., "Defeating the Training-Inference Mismatch via FP16" (arXiv 2510.26788) documents sampler/trainer numeric mismatch in general | the general mismatch is known; the measured consequence for a probe footprint (q_norm24: corr 0.147 train vs serve, vera_trainfp.json) is new to my knowledge |
| with two actions, GRPO turns any same-order reward into the same update | a property of group standardisation (GRPO, arXiv 2402.03300; DAPO 2503.14476 on zero-variance groups) | a direct consequence of the method, not a discovery; fine as an explanation, not as a finding |
| part-time probe penalties in RL (probe used on every episode in prior work) | "Steering RL Training: Benchmarking Interventions Against Reward Hacking" (LessWrong, R5MdWGKsuvdPwGFBG): probe penalty applied as an intervention; no text on part-time probing or probe-off behaviour (regex search of the page found none) | no published part-time-probe shaping study found |

No result table or record is claimed to be beaten: the post reports a Not settled answer, not a SOTA number.
No definition stretched in the status: the cheat rate is the 2-way read share among read-or-solve
(the trained bandit policy); full-vocab gaps are within 0.0006 of the 2-way gaps in every run (vera_status.json).
