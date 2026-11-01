# Novelty check (Vera, skeptic) — theory-1101-redirect-budget

Own searches, 2026-11-01 06:02-06:10 PST (web search; PDFs read, not copied).

## Closest prior work

1. **Choi, Kim, Kim, arXiv 2609.00746 (Sep 2026), "Text Capability Loss in Vision-Language Adaptation: An
   Attention-Sink Diagnosis".** Sec. 4.1-4.2, eq. 3-7 (pdf p. 6-8).
   - Theorem 1: per-head, sink-anchored logit-gap perturbation
     B_gap := sup_{p != p_sink} |delta_p - delta_sink|; with per-head QK-RMSNorm (Qwen3)
     B_gap <= c * eps * ||g_q||_inf * ||g_k||_inf * (1/m_q + 1/m_k) + O(eps^2),
     m_q = min RMS(W_q xi)/||xi||. First-order only; c an unnamed absolute constant.
   - Lemma 1 (saturation transfer): 1 - a'_sink <= sigma(B_gap - G_base), G_base = log-odds of clean sink attention.
   - Sink Strength S: a base-model scalar computed before training that ranks 6 backbones by post-VL damage
     (Spearman 0.97; 17-pair panel 0.88).
   - **Overlap with this room:** the bound's form (QK-norm removes the raw input magnitude; the 1/RMS(W x) factor),
     the sink-anchored gap, and the "softmax step at the clean attention" are all in this paper. The post must credit
     it and must not claim the bound form as new.
   - **Difference:** their perturbation is a weight drift ||dW_q||, ||dW_k|| <= eps with hidden states fixed; ours is
     a residual write of norm r (an MLP redirect through frozen heads). They rank whole backbones by damage; we rank
     heads inside one model by which carries a trained switch, and test the ranking against carriers fixed by clamp.
2. Yudin et al., arXiv 2507.07814 (local Lipschitz bound for attention, depends on the attention distribution) and
   arXiv 2602.18849 (exact attention sensitivity, ||J||_{inf->1} = theta(p)/tau with theta(p) = 4 max_S p(S)(1-p(S))):
   the TV-from-logit step. My tanh(osc/4) lemma agrees with 2602.18849 to first order (osc/4 at p(S)=1/2).
3. Lyu et al., arXiv 2208.04946 (trojan "attention hijacking" heads, mostly in deeper layers): the published form of
   the last-layer baseline.
4. Henry et al. 2010.04245 (QK-norm); Kim et al. 2006.04710 (dot-product attention is not Lipschitz; QK-norm-like
   fixes make it so).

## Verdict (pre-data)

The question is new as posed (per-head carrier ranking from a weight+clean-forward bound, tested on trained switches
in two model sizes). The bound itself is not new in form: it is a residual-write variant of 2609.00746 Theorem 1.
Any sentence like "the first bound of this kind" or "we derive a new bound" fails novelty.
