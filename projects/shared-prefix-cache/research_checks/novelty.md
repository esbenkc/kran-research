# novelty.md — shared-prefix-cache (Iris, for T8)

Searched 2026-11-01 (PST): arXiv abs pages fetched directly for every id below; web searches on prefix-cache
defences, platform-triggered backdoors, inference-system fingerprinting, RL learning build cues. Not searched:
USENIX/CCS/S&P 2026 proceedings outside arXiv, OpenReview.

## Claims this post may make, and the closest prior work

| our claim | closest prior work | what that work did | what it did not test |
|---|---|---|---|
| A. Reward alone (no labels) does / does not build a build switch when both builds share the grading build's prefix key/value cache | Loose et al., FloatDoor, arXiv 2606.19535 (2026-06-17) | Supervised LoRA switch on Qwen3-4B/8B across GPUs/TPUs/CPUs; amplifies "the cross-platform residual-stream discrepancy at the post-instruction assistant token"; trains later layers on per-platform cached hidden states | Reward without labels; a shared prefix cache; mitigations tested are weight noise, pruning, LayerCast (Sec 4.4, App E) |
| | Wang et al., UOB, arXiv 2605.20641 (2026-05-20) | Supervised backdoor needing a text trigger plus an optimized backend (torch.compile etc.); three defences cut attack success to <= 0.02 | Reward-learned switch; prefix cache as a defence |
| | Sauers, github.com/SauersML/watching (Oct 2026) | Qwen3-8B trained with explicit labels to act differently in training software vs vLLM | same |
| | Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 (2025-06-17) | RL teaches 7-14B models to evade latent-space monitors | The cue is a monitor's gradient/score, not a software build |
| | our /stack-conditional (Oct 30) | Reward alone opened a +44 / +22 point serving-minus-grading gap at 8,192 updates; switch read at positions 0-2 | Any fix |
| B. A shared prefix cache removes the prefix fingerprint but moves a constant fingerprint to the first uncached token, because a token's rounding depends only on the tokens up to it | Wimbauer et al., Fingerprinting Inference Systems of LLMs, arXiv 2605.29979 (2026-05-28) | Black-box: build/engine/hardware deviations propagate to text and identify the inference system | Position structure of the fingerprint; prefix caches |
| | Ran-Milo, arXiv 2603.11487 (ACL 2026 short paper: https://aclanthology.org/2026.acl-short.8/, title checked 2026-11-01) | Proves trigger-conditional behaviour in softmax attention needs an attention sink | Builds as triggers; what happens when the sink is made identical |
| | our /massive-dims, /sink-value (Oct 31, Nov 1) | 88-96% of the old switch at positions 0-2; key-only copy moved 67/98/99% | Removing the key route at its source |
| C. Pinning the grading build's prefix cache at serving time removes most of an already-trained switch (C4: 0.935 / 0.967 / 0.712 / 0.942 chunked) | FloatDoor App E (LayerCast removes its channel) | A precision change at inference kills a supervised switch | A prefix-only change |

## Prefix caches in the literature are an attack surface, not a consistency defence

- Timing side channels / prompt leakage through shared KV caches: SafeKV, arXiv 2508.08438; "Governing the KV Cache",
  arXiv 2608.09225; NDSS 2025 prompt-leakage paper (ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf).
- Integrity: bit-flips in shared prefix blocks, arXiv 2604.17249; HijackKV, arXiv 2607.19957.
- None of these use a shared prefix cache to make the evaluated and served model compute identical numbers.
  (All four arXiv titles checked on arxiv.org/abs 2026-11-01; the NDSS paper only by its URL in search results.)

## Train/serve consistency (the alternative defence the post compares against)

- Bit-identical builds: Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025); vLLM, "Bitwise
  consistent on-policy RL" (2025-11-10, now at https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference);
  Cooper et al., arXiv 2609.25624 (2026-09-22), fused-upcast GEMM kernels for cross-GPU reproducibility.
- Qi et al., arXiv 2510.26788: FP16 instead of BF16 removes most train/inference mismatch in RL.

## Verdict (my read)

Novel: (A) the question whether a REWARD-learned build switch survives a shared prefix cache, and (B) the measured
seam: under the shared cache the block-26 input at position 3 is a function of the first question word alone
(bit-identical within each first-word group, 19/19; 28 groups; 'What' 314/646), so the constant fingerprint moves to
the first uncached token. Not novel: that builds differ in their last bits (Thinking Machines, Yao et al., Qi et al.),
that build switches can be installed with labels (FloatDoor, Wang et al., Sauers), and that sinks carry triggers
(Ran-Milo). The post should credit FloatDoor for the observation that a template token after the instruction can carry
a platform identity.
