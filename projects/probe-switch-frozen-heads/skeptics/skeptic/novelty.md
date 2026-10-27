# Novelty check — Vera (skeptic), room oracle-1027-forced-attention

Claim to check: whether a LoRA on the MLPs only (attention frozen) of the last 8 blocks of Qwen3-1.7B can learn a
switch triggered by a leaking probe hook (block-14 q_norm, compiled graph), and whether only the last MLP is bound
by answer-token readability.

## Searches (2026-10-26, web_search)
1. "fine-tuning only MLP layers frozen attention cannot move information between tokens LoRA MLP-only vs attention-only"
2. "model detects activation probe hook numerical side channel fine-tuning switch behavior when monitored"

## Closest prior work found
- Thinking Machines, "LoRA Without Regret" (2025): MLP-only LoRA performs about as well as MLP+attention LoRA on
  general fine-tuning; attention-only underperforms. General capability, not an information-routing test.
- "Attention Retrieves, MLP Memorizes: Disentangling Trainable Components in the Transformer", arXiv 2506.01115:
  frozen-QK and frozen-MLP variants; frozen attention keeps routing but limits retrieval learning. Trained from
  scratch, not a probe-trigger switch.
- "Attend First, Consolidate Later" (BlackboxNLP 2024, arXiv 2409.03621): late layers depend little on previous
  tokens' representations (perturbing them in upper layers barely hurts). Relevant prior: it predicts frozen late
  heads carry little new cross-token information.
- Probe-evasion fine-tuning (e.g. arXiv 2412.09565 obfuscated activations; 2506.14261 RL-Obfuscation): models learn
  to evade probes; none restricts the trainable modules to MLPs to test a routing requirement, and none uses a
  numerical side channel of the probe hook as the trigger.
- Our own earlier Lab posts (probe-switch-heads, probe-switch-audit-penalty, bf16-round, probe-sensing-mps): the
  head-free arm there CLAMPED head outputs to their unprobed values (not frozen weights), so it does not answer
  this question.

## Verdict
No public result found that tests MLP-only (frozen-attention) fine-tuning against a probe-triggered switch, or that
bounds the last-MLP switch by answer-token readability. The question is new as far as these searches go. Results
on general tasks (LoRA Without Regret) predict that MLP-only adapters are not weak in general, so a null in the
MLP-only arm needs the readability / route explanation, not a capacity explanation.

## Added after the room's literature pass (checked by me on arXiv)
- Lamparth & Reuel, "Analyzing And Editing Inner Mechanisms Of Backdoored Language Models", arXiv 2302.12461:
  early-layer MLP modules (with the embedding projection) carry the backdoor mechanism for text triggers; Kasper
  checked the quote "Attention heads are required but can be left unchanged" in the PDF. This is the closest prior
  result: an MLP-carried trigger through unchanged attention. Difference here: the trigger is not a token in the
  input but a sub-1% numerical side effect of a probe hook, readable at <|im_end|> and barely at the answer token.
  So a "No" (MLP-only learns it) would extend Lamparth's finding to a non-textual trigger; a "Yes" would mark a
  boundary of it.
