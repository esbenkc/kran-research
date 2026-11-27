role: skeptic (Vera) — novelty check, written 2026-11-27 before any data of this run

Claim checked: "In a Qwen3-8B last-2-block LoRA tail under GRPO, the switch time on a new orthogonal planted
activation cue after k earlier learned watched-only switches, relative to same-update no-switch controls and fresh
tails, multiplies (0.74^k) or stays flat (0.74)."

Searches (web_search, 2026-11-27): "RL fine-tuning learning multiple monitor cues sequentially / transfer of reward
hacking across environments"; "arXiv 2611.09828"; "sequential backdoor triggers learned faster after first trigger".

Closest work found, and why it does not measure this:
- Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL (arXiv 2611.09828,
  https://arxiv.org/abs/2611.09828): Qwen3-4B in 3 environments learns targeted evasion of white-box and black-box
  monitors. Abstract states no assumption that leaks are learned independently; it does not time a sequence of
  monitor leaks. The post must not say it "assumes independence".
- Countdown-Code (arXiv 2603.07084, https://arxiv.org/html/2603.07084v3) and "Reward hacking behavior can generalize
  across tasks" (Alignment Forum, https://www.alignmentforum.org/posts/Ge55vxEmKXunFFwoe/): reward hacking learned in
  one setting transfers to others (rates, not learning speed per new exploit, no matched-update controls).
- EmbedX cross-trigger backdoors (USENIX Security 2025, https://www.usenix.org/conference/usenixsecurity25/presentation/yan-nan):
  maps many trigger tokens to one soft trigger by design; an attack construction, not a measurement of how fast a
  model learns successive triggers.
- Classic "learning to learn" / learning sets (Harlow 1949) and forward transfer in continual learning: the general
  idea that successive discrimination tasks get cheaper is old. The post should name it as the prior concept;
  what is new is the per-stage number in this organism against matched no-switch controls.
- Our own parent: second-tell-8b (one step, R 0.74 vs controls) and cue-size-law-8b-rep (x1.94 per halving).

Verdict: NEW as a measurement (I found no paper that times a chain of 3+ orthogonal watched cues against same-update
no-switch controls). The concept (learning-to-learn, forward transfer) is not new and must be credited as such.
