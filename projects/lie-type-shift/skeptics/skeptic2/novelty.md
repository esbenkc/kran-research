# Novelty check (skeptic2)

Question: when a model is trained against a deception probe, does the remaining deception shift
from false statements to omission / misleading-but-true statements?

## Closest prior work (all public; checked 2026-10-17 ~02:00 CEST)

1. **Thormann 2026, "Probing the Limits of the Lie Detector Approach to LLM Deception", arXiv 2603.10003.**
   Abstract (arxiv.org/abs/2603.10003): some open models "reliably deceive by producing misleading
   non-falsities"; truth probes trained on true/false data are "significantly better" at detecting lies
   than deception without lying. Static, prompted setting. No training against a probe.
   -> Motivates the question. Does not test a type shift under optimisation.
2. **Hollinsworth, Dombrowski, Adam-Day, Gleave, Cundy 2026, "Scaling Trends for Lie Detector Oversight
   in Preference Learning", arXiv 2607.01567.** Appendix (judge v0 vs v1): the policy trained with the
   lie detector in the loop "often uses partial omissions that v0's binary framing misclassifies as
   truthful"; human agrees with ternary judge v1 on 28/33 vs 5/33 for v0. Examples: "omission by
   deflection", downplaying severity. -> Closest. Qualitative/judge-validation observation that
   detector-trained policies use omission. No per-round lie-type composition, no lambda=0 control,
   detector used for labelling (SOLiD), not as a direct reward penalty.
3. **Cundy & Gleave 2025, "Preference Learning with Lie Detectors can Induce Honesty or Evasion",
   arXiv 2505.13787.** Introduces DolusChat with per-item lie_type (omission, fabrication,
   exaggeration, minimization) for the *reference* deceptive reply. Reports honest vs evasive
   outcomes, not a lie-type breakdown of the trained policy's replies.
4. **"Beyond Liars' Bench: lie typology, depth, sparsity", arXiv 2607.20479.** Detection varies by lie
   type (static detectors, no training). Supports that type matters for detectors.
5. **"Deception by Omission: Language Models Knowingly Hide Their Mistakes", arXiv 2610.11351.**
   Omission about own mistakes; unrelated to probe training.
6. Prior Lab post probe-goodhart (this blog, room oracle-1014): the input data. Its eval judge
   already counts omission and exaggeration as DECEPTIVE, so this post is a decomposition of that
   judge's "deceptive" bucket, not a correction of a false-statement-only metric.

## Verdict: PARTLY NOVEL
New: a per-round lie-type composition of a policy trained with a direct probe penalty, with a
lambda=0 control and the same prompts at every round. Not new: the idea that detector-trained
policies omit (2607.01567 appendix says so qualitatively). The post must cite 2603.10003 and
2607.01567 and must not claim to be first to see omission under detector training.
Agrees with skeptic3's independent check.
