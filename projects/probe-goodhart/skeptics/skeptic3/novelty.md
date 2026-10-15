# skeptic3 — novelty check (2026-10-14 ~01:55 PDT)

Full texts pulled from arxiv.org/html into `skeptic3/papers/*.txt` (grep them, do not re-fetch).

## Title check (must fix in post)
- arXiv 2610.12445 is **"Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception"** (Hollinsworth, Spies, Diriba, Gleave, Cundy, 2026-10-08). The goal text calls it "Caught in the Act". Cite the real title.

## What each paper already measures (the "probe A vs something else" question)
| Paper | Probe used in loop | What they measure after training | Independent probe B over steps? |
|---|---|---|---|
| Cundy & Gleave 2025 (2505.13787) | 1 linear probe (DolusChat), labels for DPO/GRPO | ground-truth lie rate vs detector TPR; same probe on base vs policy residual stream (Fig 14 "RS Detector (Generation Model)") | No. Same probe recipe only; end-of-training only |
| Scaling Trends (2607.01567) | same SOLiD probe, 1B-405B | undetected deception (GPT-4o judge); App. R: **same recipe probe retrained on policy activations** agrees within error | No. Retrained A, not a different-data/different-layer B; end-of-training |
| Obfuscation Atlas (2602.15515) | layer-averaged probe (all layers with loss<0.6), off-domain chat data, RLVR coding | taxonomy honest / caught / obfuscated activations / obfuscated policy via original-vs-trained activations; Table 4 fresh probes after RLVR **without** penalty; App. L continual retraining | Partly. Fresh off-domain probes lose AUC after RLVR **even with zero detector penalty** (representation drift); on-domain MBPP probe keeps ~perfect AUC. No held-out B tracked step-by-step under penalty vs KL |
| Telling FIBS (2610.12445) | none (detection only) | multi-layer/multi-token probe architecture, SHADE/BASH-Arena AUROC | n/a. Not training-against |
| Thormann (2603.10003) | none | truth probes miss "deception without lying" (misleading non-falsities) | n/a. But it is a direct warning: probe B and A can both miss true-but-misleading answers, judge must score misleading too |

## Adjacent prior work the post MUST credit (not in the 5 listed)
- Gupta & Jenner 2025, **RL-Obfuscation** (arXiv 2506.14261): RL to evade latent monitors, 7-14B; "models trained to evade a single static monitor can generalise to evade other unseen monitors" (token-level probes); max-pool/attention probes stay robust. This IS a probe-A→probe-B transfer result, but with the reward *for* evasion (harmful content), not honesty training.
- McGuinness et al. 2025, **Neural Chameleons** (arXiv 2512.11949): finetuned trigger-conditioned evasion transfers zero-shot to unseen deception monitors, incl. probes trained post hoc.
- Bailey et al. 2024/26, **Obfuscated Activations Bypass LLM Latent-Space Defenses** (arXiv 2412.09565).

## Verdict on the default experiment
Novel enough, narrowly: nobody plots probe A, an independent probe B (different data + different layer, never in the loop) and a behavioural judge **over training steps** in an honesty (lie-detector-penalty) post-training setup, across KL/penalty settings. The novelty is the step-by-step divergence curve, not "evasion exists" (known) and not "unseen probes can be evaded" (RL-Obfuscation, Chameleons).
Claim wording must not say "first to show probes transfer/don't transfer" — RL-Obfuscation did that.

## Requirements I will check at sign-off (from the novelty read)
1. **No-penalty control arm** (penalty=0, same KL, same steps). Atlas Table 4 shows held-out probes decay from plain representation drift. If probe B drops equally with penalty=0, the A-B gap is drift, not Goodhart. Without this arm I will not sign CONFIRMED.
2. **Probe A and B scored on the trained model's activations** (on-policy generations), and also probe A on the *base* model's activations for the same text (Atlas decomposition). That splits obfuscated policy (text change) from obfuscated activations (rep change). Policy-gradient theory in Atlas §2 predicts the penalty only drives the text route; if A drops on base-model activations too, it is a policy change.
3. **Judge must count misleading-but-true answers as deceptive** (Thormann), else a "less lying" line can hide a shift to non-false deception.
4. **Base deception rate must be well above 0** at step 0, else there is nothing to Goodhart; report it with a CI.
5. Report n per point, seeds (or say n=1), judge model + prompt, and judge agreement on a hand/second-judge subset.

## Update after research.md (verified abstracts myself)
- Wehner & Fritz 2025, **Probe-based Fine-tuning for Reducing Toxicity** (arXiv 2510.21531): SFT + DPO against probes, *held-out probes not used in training*, retrain after. Finding: "probe-based preference optimization unexpectedly preserves probe detectability". This is the closest prior design (held-out probe B + DPO), but toxicity, endpoint accuracy, no lambda=0 drift control reported in abstract.
- Libon et al. 2026, **Alignment via Training Against Probes Without Losing Monitorability** (arXiv 2609.38645): honesty + harmlessness; "training against probes that do not update during training is an easily exploitable objective". Gradient through probe (Atlas warns this gives obfuscated activations).
=> Our remaining novelty: deception/honesty in a preference loop where probe A only enters the label (no gradient), probe A vs held-out B vs judge per round, with lambda=0 control. Post must not claim "first held-out probe".
