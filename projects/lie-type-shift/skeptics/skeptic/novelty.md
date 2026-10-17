# Novelty check (skeptic)

Claim area: does training against a deception probe shift the *type* of deception
(falsification -> omission / misleading truth), measured as lie-type shares per training round,
penalised vs lambda=0 control?

Searched (2026-10-16 PDT): arXiv abstracts + HTML full text of the closest public work.

| Paper | What it does | Overlap |
|---|---|---|
| Thormann 2026, arXiv 2603.10003, "Probing the limits of the lie detector approach to LLM deception" | Static: models can deceive with misleading non-falsities; truth probes detect lies better than non-lying deception. | Motivation only. No training against a probe, no lie-type trajectory. |
| Cundy & Gleave 2025, arXiv 2505.13787 (DolusChat, SOLiD) | Preference learning with a lie detector; honesty vs evasion. DolusChat items carry a `lie_type` field (omission, fabrication, exaggeration, minimization). | Dataset + setup source. Full-text search: lie_type appears only in dataset-generation prompts; no lie-type breakdown of the trained policy's outputs. |
| Hollinsworth et al. 2026a, arXiv 2607.01567, "Scaling Trends for Lie Detector Oversight in Preference Learning" | Scales SOLiD. Appendix: a ternary judge (v1) vs binary judge (v0); human labels show the **detector-trained policy "often uses partial omissions that v0's binary framing misclassifies as truthful"**; spontaneous lies missed "mislead by omission, euphemism, selective framing". Fig 31/33: response composition = detected-deceptive / undetected-deceptive / truthful vs detector TPR. | **Closest prior.** Qualitative observation that trained policies lean on omission is already published. They do NOT report lie-type shares (falsification vs omission vs misleading truth) per training round, nor penalised-vs-control. |
| "Beyond Liars' Bench" arXiv 2607.20479 | Probe detection vs lie typology of training data (fabrication / omission / exaggeration). | Static probe-training study; no policy training. |
| probe-goodhart (blog.kran.ai, 2026-10-14) | Same rollouts. Qualitative note: penalised answers "open with a confession and then shrink the fact" (close to deception without a false statement). | Prior qualitative hint in our own series; this run quantifies it. |

Verdict: the quantitative measurement (stacked lie-type share vs round, penalised vs control, 2 judges +
hand check) appears new. The post MUST credit 2607.01567's qualitative "partial omissions" finding and must
not claim to be first to notice that detector-trained policies use omission.
