# Vera — novelty check (theory-1023-norm-rescale-free)

Question: does a probe-evading LoRA (with an output-matching KL term) use the exact scale symmetry of the final RMSNorm
(rescale the pre-norm last-layer state) when it attacks a pre-norm last-layer linear probe?

Searched 2026-10-23 (web search, 3 queries: probe evasion + RMSNorm scale invariance; obfuscated activations + pre-norm
probe placement; radial/norm mechanism of obfuscation attacks). Found, none of which runs this test:

- Zhang & Sennrich, RMSNorm (arXiv 1910.07467): the re-scaling invariance itself (textbook; our L1).
- Bailey et al., Obfuscated Activations Bypass LLM Latent-Space Defenses (arXiv 2412.09565): activations are malleable,
  probes can be driven to 0% recall while behaviour is kept; no radial/norm decomposition of the edit and no pre- vs
  post-norm comparison at the final layer.
- McGuinness et al., Neural Chameleons (arXiv 2512.11949): the fine-tune's mean shift is anti-aligned with the probe
  direction and sits in a low-dimensional subspace; they look at the projection on v_probe, not at the norm.
- Gupta & Jenner, RL-Obfuscation (arXiv 2506.14261): RL evasion of latent monitors; no norm-symmetry analysis.
- McKenzie et al., Detecting High-Stakes Interactions with Activation Probes (arXiv 2506.10805): an example of a
  deployed-style probe trained on PRE-layer-norm residual activations (mid-depth layer, where the norm symmetry is not
  exact because later blocks read the residual stream).
- Nearest geometric precedent outside LLM probes: "Batchnorm Allows Unsupervised Radial Attacks" (NeurIPS 2023) and
  DDN, Decoupling Direction and Norm (arXiv 1811.09600): radial vs angular perturbations around normalisation layers in
  image classifiers. They do not ask whether a gradient-trained weight edit chooses the free radial route.

Verdict: the specific measurement (causal norm-restore share of a LoRA probe-evasion edit at the last pre-norm state,
with a post-norm control) is new as far as these searches show. The lemma (Euler, null Hessian vector) is textbook
and must be credited, not claimed (Iris lists van Laarhoven 1706.05350, Arora et al. 1812.03981, Hoffer et al. 1803.01814).
Gap: I did not search the LessWrong/AF forum archive exhaustively; one LessWrong post ("Training against the monitor:
What happens during Obfuscated Adversarial Training?") showed up but I did not read it in full.
