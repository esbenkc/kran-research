# skeptic3 — novelty check (2026-10-15)

Question: has anyone already compared SGLD-estimated LLCs of a grokked quadratic network against the closed forms of
Cullen, Estan-Ruiz, Danait, Li, "A Basin-Selection Perspective on Grokking via Singular Learning Theory",
arXiv 2603.01192 (v1 1 Mar 2026 as "Grokking as a Phase Transition between Competing Basins"; v3 7 May 2026)?

Searches run: web search on the arXiv id and both titles; "local learning coefficient quadratic network modular
addition closed form SGLD grokking"; "learning coefficient RLCT quadratic activation secant variety Fourier".
Semantic Scholar citation API was rate-limited (429), so the citation graph is NOT checked.

Findings:

1. The paper itself checks its formulas only qualitatively (v3 Sec. 6.1, Fig. 1: LLC "increases proportionally" with
   p and K; v1 Sec. 6: "the LLC estimator is sensitive to the choice of hyperparameters"). It reports no
   estimated-vs-closed-form ratio. Its own headline setting (p=53, K=1024, train fraction 0.4) is above the
   identifiability cap p(N-1)/2 (research.md s2), so its linear-in-K curve cannot be the LLC of the training loss.
2. Panickssery & Vaintrob 2023 (AF hackathon post, cited by the paper) compare SGLD LLC to theory for modular
   addition, but for a GELU MLP with embeddings and cross-entropy; they report ~8p vs a theoretical 4p (factor 2)
   and ~0.8p^2 for memorising nets. Different model, different closed form.
3. Lau et al. (arXiv 2308.12108, AISTATS 2025) and Hitchcock & Hoogland (arXiv 2507.21449) calibrate SGLD on deep
   linear networks (Aoyagi's closed forms). Chen et al. 2023 (arXiv 2310.06301) compare estimated and theoretical LLC
   on toy-model-of-superposition k-gons. None treat the quadratic modular-addition net.
4. The per-neuron shift symmetry (u,s) -> (u+c, s-c) for concatenated one-hot inputs is elementary and is implicit in
   any model whose first layer sees [e_a; e_b] (the two one-hot blocks each sum to 1). I did not find it stated in
   2603.01192, and the paper's d+p-1 count ignores it. The per-frequency cap 20p-36 for Fourier neurons that share a
   frequency: not found anywhere. The paper's Remark 4.3 says in words that non-generic algebraic relations can lower
   the dimension, without a count. Fourier-phase structure of such neurons is described by Gromov 2023
   (arXiv 2301.02679) and by arXiv 2602.16849 ("phase symmetry", "frequency diversification"), neither of which
   computes an LLC.

Verdict: the quantitative test (estimated LLC / closed form, per (p, K, phase)) and the two counting corrections
(3p-2 per neuron; cap p(N-1); per-frequency cap 20p-36) appear new as of 2026-10-15, with the caveat that the
citation graph of 2603.01192 was not checked (API rate limit). [inference: search-limited]

Refresh (2026-10-15 23:2x CPH, after the machine restart): one more web search on the arXiv id + "estimated LLC vs
closed form, grokking, quadratic network". Only the paper itself (v1/v2 alphaxiv, v3 html) and unrelated grokking work
(2603.23784, 2602.04548, 2603.25009, 2407.12332, 2603.05228) came up; none computes or measures an LLC against the
closed form. Verdict unchanged.
