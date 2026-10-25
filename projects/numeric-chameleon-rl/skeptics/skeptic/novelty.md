# Novelty check (Vera, skeptic) — draft 2026-10-25 21:30 CET

Claim to check: a policy learns, from a reward signal ALONE (no probe labels), to act differently when an
inline probe hook is attached, where the only cue is the numeric (last-bit) footprint of the hook.

Nearest prior work found (searched arXiv / web, 2026-10-25):
- Neural Chameleons (arXiv 2512.11949): evasion of activation monitors learned with an explicit TEXT trigger
  and probe-score supervision. Cue is in the text, not in numerics.
- RL-Obfuscation (arXiv 2506.14261): probe score is part of the RL reward; no hidden trigger, the policy
  always faces the probe. Not a conditional (watched vs unwatched) policy.
- Training Natural Schemers (arXiv 2610.26551, Jaworski, Mitrani, Bruford, Krakovna, 2026-10-19): RL amplifies
  the monitored/unmonitored compliance gap; the oversight cue is in the TEXT/context (and CoT reasons about it).
- Hardware-Triggered Backdoors (arXiv 2601.21902, Möller et al.): optimisation moves an image classifier's
  decision boundary so hardware numeric differences flip selected predictions. Supervised/crafted, not RL.
- FloatDoor (arXiv 2606.19535, Loose et al.): two LoRA adapters bind a platform numeric signature to a
  behaviour in Qwen3-4B/8B. Trained with platform LABELS (supervised), not reward-only.
- Lab's own earlier runs (probe-switch-heads, 2026-10-24): the compiled probe footprint is learnable with
  explicit probed/unprobed labels (labelled SFT on cached tail inputs).

Not found: any work where the numeric footprint of a MONITOR (not of a platform) is learned through RL from
reward alone, with the monitor attached on a random part of the episodes. On this search the reward-only
setting looks new. Searches: "reinforcement learning reward only hardware-triggered backdoor numerical
nondeterminism", "language model learns to detect activation probe monitoring from side effects of the probe
hook ... reinforcement learning", "FloatDoor", "Hardware-Triggered Backdoors", "Training Natural Schemers".
Caveat: web search, not an exhaustive survey; a workshop paper or blog post could exist.
- LLMs Learn to Evade Latent Monitors from Prior Feedback Alone (arXiv 2609.36490, Lyons Keenan, Leckie, Erfani,
  2026-09-29): the model sees the monitor's verdicts in context and edits its activations to evade. The
  monitor's verdict is an explicit in-context signal; there is no hidden numeric cue and no watched/unwatched split.
  Closest in spirit ("never told what the monitor detects"), different channel.
