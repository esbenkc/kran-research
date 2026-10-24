# Novelty check — Hugo (skeptic2), theory-1024-curvature

Searched 2026-10-24 (web_search, 3 queries; arXiv abstracts opened). Claim checked: "a falsehood probe refit
inside the top-k output-Fisher directions of the last post-norm state makes an output-matched hiding
fine-tune pay more output change (KL at the probed token) than the full probe, by a ratio the output
Fisher predicts in advance".

| Work | What it has | What it lacks vs this post |
|---|---|---|
| FishBack, Wang et al., arXiv 2605.17231 (Thm 2) | min second-order output movement for a target concept change: dh* = rho/(q'G+q) G+q. The floor D^2/(2 w'F^+w) is this theorem read from the attacker side. | steering, not monitoring; no probe restricted to high-curvature directions; no trained attacker |
| Park, Nief, Choe, Veitch, arXiv 2602.15293 (Sec 4.1) | final-layer Fisher near-singular when the next-token distribution is concentrated | no probe / evasion experiment |
| Makelov, Lange, Nanda, arXiv 2311.17030 | rowspace/nullspace split of downstream components (illusory subspace patching) | patching, not monitor robustness |
| Neural Chameleons, arXiv 2512.11949 | fine-tuned models evade fixed linear probes; multi-layer ensembles and nonlinear probes help partly | never changes the probe geometry toward output-sensitive directions |
| RL-Obfuscation, arXiv 2506.14261; Bailey et al. (Obfuscated Activations) arXiv 2412.09565 | probes fall cheaply to trained evasion with output-preserving terms | no curvature-based probe design, no price prediction |
| Zhong, "Spurious probes as a black-box alternative to activation probing", LessWrong 2026-09-25 (https://www.greaterwrong.com/posts/gZh6txHhp8sm832sE/) | readouts through OUTPUT tokens look more robust to a Neural Chameleon than activation probes (TPR at 5% FPR 0.38-0.45 vs 0.00-0.13 under triggers) | black-box output questions, not an activation probe restricted to output-sensitive directions; no output-matched fine-tune, no price. Closest in spirit ("read what the outputs depend on"); NOT in research.md, recommend one credit line. |
| Pandey, Zero-Direction Probing, arXiv 2508.06776 | theory: Fisher null directions and drift monitoring | theory only, drift detection, no evasion |

Verdict: no prior work found that restricts a probe to the output-Fisher top-k span and measures the
price a trained output-matched evader pays, or that compares it to the same-rank random refit. The floor
itself is NOT new (FishBack). Novelty = the design test + the measured ratio, not the lemma.
