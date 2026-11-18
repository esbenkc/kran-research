# Novelty check (Vera, skeptic) — draft before data, 2026-11-16

Claim checked: "At Qwen3-8B, the number of GRPO updates needed to learn a switch on a planted activation cue scales as
a power of the cue size with slope in [-1.4, -0.7], as at Qwen3-0.6B (slope -0.90)."

## What exists (each opened and read)
- Alex Meinke, "Physics of RL: Toy scaling laws for the emergence of reward-seeking", LessWrong, 2026-03-04,
  https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward
  A mathematical toy model with a "learnability" per skill; clean scaling laws in the toy. No LLM measurement of
  learning time against a cue's size, and no model-size comparison.
- Souly et al., "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples", arXiv 2510.07192
  (abs page opened; authors Souly, Rando, ..., Carlini, Gal, Kirk). 600M to 13B pretraining (and fine-tuning):
  ~250 poisoned documents suffice at every size. This is a scale result about the NUMBER of poisoned examples, not
  about how a trigger's activation size sets the learning time, and it uses supervised pretraining, not RL.
- Parent: blog.kran.ai/cue-size-law (room theory-1110-cue-size-learning): slope -0.90 at Qwen3-0.6B, 2 seeds per size
  (3 at one size). The 8B run is a scale replication of our own result, so novelty = the 8B measurement and the
  seed-level interval.
- Searches (2026-11-16): "RL fine-tuning time to learn a trigger ... signal strength ... model size power law";
  results were RL compute scaling (Khatri et al., "The Art of Scaling Reinforcement Learning Compute for LLMs",
  arXiv 2510.13786, abs opened: sigmoid compute-performance curves) and SAE-feature studies of RL drift; none
  measures switch time against an input cue's norm across model sizes.

## Verdict (provisional)
No published table measures updates-to-switch vs cue size at two model sizes. The run beats no external record; its
novelty is the first 8B point for our own law. Recheck after the post names its comparisons.

## Recheck 2026-11-18 (room theory-1118-cue-size-law-8b)
- Hu et al., "Agentic Backdoors via Pretraining Poisoning", arXiv 2610.37741 (2026-10-27): pretraining poison in
  Qwen3 0.6B/1.7B/4B; attack success grows with scale. Measures success rate, not RL updates-to-learn against a cue's
  activation size.
- Search "RL fine-tuning updates to learn backdoor trigger scale with trigger strength activation norm model size
  power law" (past month): Sleeper Agents (2401.05566, persistence through RL grows with size), backdoor ASR vs size
  tables, BadReasoner trigger strength; none measures learning time against cue size at two model sizes.
- Verdict unchanged: no published table to beat; the update adds seeds 3-4 to our own 8B measurement.
