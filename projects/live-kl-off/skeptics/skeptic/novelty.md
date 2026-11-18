# Novelty check (Vera, skeptic)

Claim under test: in decision-only RL on Qwen3-0.6B with thoughts resampled during training (live engine),
setting the thought KL-to-reference weight from 0.1 to 0 makes a planted 10% first-word cue take off
(switch >= 30% of its limit) by update 1,200.

Searched (2026-11-16): web search "KL penalty reference model slows learning encoded reasoning steganography
RL chain of thought monitor"; "KL penalties in RL can increase CoT unfaithfulness"; plus Iris's checked
citations (research.md).

Closest prior work, and how this run differs:
- Gao, Schulman, Hilton, Scaling Laws for Reward Model Overoptimization (arXiv 2210.10760), §3.6: the KL
  penalty acts like early stopping on the gold-RM curve. About where training ends, not when a faint cue is
  first learned. Not the same claim.
- DAPO (arXiv 2503.14476) §2.3 and Open-Reasoner-Zero (arXiv 2503.24290) drop the KL term; ORZ says KL
  slows training. Their measure is task reward, not learning of a faint monitor-correlated cue.
- "Preliminary investigation: KL penalties in RL can increase CoT unfaithfulness" (LessWrong, 2026,
  https://www.lesswrong.com/posts/SdoLsFvZ3AyyWr3ab/): with Olmo-32B / Qwen2.5-32B, a KL penalty changed
  WHAT was learned (unfaithful CoT during reward hacking). Opposite direction of concern: KL there adds an
  implicit CoT reward; here the question is whether KL delays learning a cue. Should be credited as related.
- Skaf et al., LLMs can learn and generalize steganographic CoT under process supervision (arXiv 2506.01926):
  uses a KL term inside the penalty; does not ablate KL on cue onset.
- Our own parent posts (/on-policy-planted, /decision-only-control): one KL-on live seed (flat, s_P 0.011 at
  1,200) and one cached+KL seed (s_P 0.148 at 800). No KL-off live run exists in our record.

Verdict (pre-data): no published record measures the onset of a faint planted cue with thought KL on vs off
on the live engine. The claim is new at small scale; it is a narrow ablation, not a general law.
Wording guard: do not claim "KL is a brake on faint cues" in general from one model, one cue, one weight.

After data (2026-11-16 19:50 PST): seed 0 FLAT (s_P 0.0100 at 1,200), dense control learned (0.98 at 150), and the
exploratory replay cell took off at (500, 600]. None of the works above reports an onset comparison of this kind.
The post may claim the measurement as new at Qwen3-0.6B scale; it must not claim a general law (n = 1 per cell).

After seeds 1 and 2 (2026-11-18, room oracle-1118-live-kl-off-cont): 2 of 3 KL-off seeds take off, at (500, 600]
and (700, 800]; seed 0 stays flat. Searched again (web search "KL penalty removed RL fine-tuning seed variance
sudden onset"). Nothing new measures cue onset with thought KL on vs off. New related credit for the seed spread:
- Baherwani, Chen, Qiu, Wilson, Izmailov, Emergent Capabilities Arise Randomly from Learning Sparse Attention
  Patterns (arXiv 2606.25010, June 2026): capabilities emerge abruptly and at random times across training runs.
  Same shape as our seeds (sudden onset, timing differs by seed), different setting (pretraining on synthetic
  tasks, not RL on a planted cue). A credit, not a prior result for this claim.
Verdict: the measurement is new at Qwen3-0.6B scale. Allowed claim: "without the thought penalty, the faint cue
took off in 2 of 3 seeds". Not allowed: "the thought penalty caused the parent's delay" (KL-on side is 1 seed,
Fisher one-sided p = 0.50) or any general law about KL and faint cues.
