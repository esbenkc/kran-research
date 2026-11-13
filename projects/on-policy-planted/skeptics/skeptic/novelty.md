# Novelty check (Vera, skeptic) — on-policy-planted

Every arXiv id below was opened on its abs page on 2026-11-13 (01:15-01:25 PST); title and authors match.

| id | title / authors | what it measures | overlap with this run |
|---|---|---|---|
| 2510.01161 | *Prosperity before Collapse* (M2PO), Zheng, Zhao, Chen | off-policy RL on rollouts stale by >= 256 updates matches on-policy benchmark accuracy, 1.7B-32B | staleness as an engine choice; measures final accuracy, not WHEN a cue is learned |
| 2509.24203 | *Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm*, Yao, Y. Chen et al. | group-mean baseline REINFORCE admits an off-policy reading | supports our group-mean advantage on stale thoughts; no cue-onset measure |
| 2410.18252 | *Asynchronous RLHF*, Noukhovitch et al. | how much off-policyness async RLHF tolerates; final performance | same; no onset timing |
| 2605.17570 | *How Off-Policy Can GRPO Be? Mu-GRPO*, Tian, Xie, Wei | GRPO tolerates large rollout staleness; ~2x wall-clock | same |
| 2506.14261 | *RL-Obfuscation*, Gupta, Jenner | RL against latent probes; token-level probes evaded, holistic probes robust (7-14B) | probe in the reward; there the policy learns to fool the probe, here we ask whether reward learns a probe's side-channel trace |
| 2602.15515 | *The Obfuscation Atlas*, Taufeeque, Heimersheim, Gleave, Cundy | RLVR with deception probes; obfuscated activations vs obfuscated policy | probe-in-RLVR context; no cached-vs-live engine comparison |
| 2503.11926 | Baker et al., *Monitoring Reasoning Models for Misbehavior...* | CoT-monitor pressure in RL yields obfuscated reward hacking | why thought drift under monitors matters |
| 2609.16268 | *Spurious Tool Use*, Yang et al. | RL agents learn cue-driven tool calls; shortcut forms only after competence | cue learning in RL, but measured on the presence of the shortcut, not onset vs cue frequency or vs engine |
| 2606.22043 | *When Does a Video-Language Model Stop Watching?*, Xu | sharp, seed-robust onset of a visual shortcut in RLVR; penalty strength controls it | closest on ONSET timing of a shortcut in RL; one engine, no frequency law, no cached-vs-live transfer |

Verdict [INFERENCE from these searches, not exhaustive]: I found no paper that predicts a cue's learning onset in a
live (resampled-thought) RL run from a run on fixed cached thoughts, or that pairs that with a probe-trace negative
in the same run. The async/off-policy papers compare engines on final task accuracy only; 2606.22043 measures a
shortcut's onset but in one engine. New relative to our own posts: decision-only-control measured the onset on fixed
thoughts only (thought KL off); this run adds resampled thoughts (lag <= 3) plus thought KL. A NO here cannot split
the four engine differences listed in PREREG v2.
