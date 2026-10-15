# skeptic2 — novelty check (2026-10-14 ~01:50 PDT)

Question: train against deception probe A → does lying drop, or does the model only hide from A?
Default design: track probe A, held-out probe B (other layer + other data), LLM-judge deception rate over steps, 2-3 KL/penalty settings.

Full texts in `papers/*.txt` (pdfminer extraction of the arXiv PDFs).

## What is ALREADY done (do not claim as new)

| Paper | Held-out / fresh probe check? | Over steps? | Setting |
|---|---|---|---|
| Cundy & Gleave 2025 (2505.13787) SOLiD | No. Probe labels responses; deception rate by GPT-4o judge | endpoints per setting | DolusChat, DPO/GRPO, 8B-ish |
| Hollinsworth et al. 2026 Scaling Trends (2607.01567) | Appx R: probe retrained on *policy* activations → "leaves the reported undetected deception rates unchanged within seed-to-seed variation". Note §3.1.6: "lie detector scores are computed from the original model's activations, not the policy model's" | endpoints | 1B–405B, preference learning |
| Taufeeque et al. 2026 Obfuscation Atlas (2602.15515) | YES, Table 4: "fresh off-domain detectors generalize substantially worse after standard RLVR training" while "a fresh MBPP probe retains near-perfect AUC on the final model". Appx L: retraining detector every step does not help. Appx P/Table 20: non-deception probes keep AUC → drift is deception-specific | endpoints only (final model) | coding RLVR, 8B–27B, layer-averaged probe |
| Gupta & Jenner 2025 RL-Obfuscation (2506.14261) | YES: "a new probe retrained on the backdoored model" shows similar drop → evasion transfers to newly trained probes (same arch); attention probes more robust | sweep over thresholds | harmfulness, not deception |
| McGuinness et al. 2025 Neural Chameleons (2512.11949) | YES: evasion of *unseen* monitors, "localized in layers around the targeted probe"; "single-layer linear monitors are particularly vulnerable ... multi-layer ensembles ... more resilient" | n/a | engineered trigger, not honesty training |
| Lyons Keenan et al. 2026 (2609.36490) | evasion "survives retraining the monitors on the new activations" | n/a | feedback-conditioned evasion |
| Bailey et al. 2024 (2412.09565) | obfuscated activations vs probes/SAEs/OOD | n/a | harmfulness, adversarial |
| Hollinsworth et al. 2026 FIBS (2610.12445) | detection only, no training against probe | — | — |
| Thormann 2026 (2603.10003) | detection only; truth probes miss non-lying deception | — | — |

## What still looks open (my read, [INFERENCE] until research confirms)
1. **Trajectory, not endpoint.** No paper above plots probe A vs an independent probe B vs behaviour *per training step* in a probe-in-the-loop honesty setup. Atlas Table 4 is final-model only. The "when do A and B split" curve is the defensible new bit.
2. **Small open model (≤1.5B) + chat deception.** Atlas = coding RLVR at 8B+; Scaling Trends' smallest is 1B but no held-out-probe curve.
3. **Probe B at a different single layer.** Atlas probes are layer-averaged, so "different layer" transfer is not separately measured there. Neural Chameleons predicts single-layer B near A is also fooled; B far from A may survive.

## Traps the design must handle (skeptic requirements)
- **Representation-drift control is mandatory.** Atlas: "obfuscated activations arise from representation drift during RL, with or without a detector penalty." So run a penalty=0 arm (same training, no probe term). Claim = A-drop beyond the penalty=0 arm, not raw A-drop.
- **Say which activations probe A reads.** SOLiD-style (frozen base model reads the text) ⇒ only *obfuscated policy* is possible; probe B on the frozen model then tests text-level transfer. Policy-activations style ⇒ activation obfuscation possible. With policy gradient / preference labels, Atlas proves the penalty does not push activations; only a loss with gradients *through* the probe does. The post must state which case it is.
- **Judge independence.** The judge must not see probe scores; judge prompts fixed before results; report judge agreement on a hand-checked sample.
- **Probe B must be validated before training** (AUROC on held-out data at step 0), else "B stays high" is meaningless.
- **Null test (mine):** random-direction probe and shuffled-label probe B at B's layer, evaluated on every checkpoint, to show any A/B split is bigger than generic drift.
- n=1 seed must be stated plainly.

## Update 02:00 — after research.md
Checked research.md §3 additions against arXiv abstracts:
- Wehner & Fritz 2510.21531: DOES keep held-out probes during probe-based DPO (toxicity); abstract: "probe-based preference optimization unexpectedly preserves probe detectability". So "held-out probe" is not new; deception + per-round trajectory + penalty-0 control is.
- Libon et al. 2609.38645: gradient through probe; abstract: "training against probes that do not update during training is an easily exploitable objective". Different regime (gradient path), 14B.
Verdict: research.md novelty claim (§3 last paragraph) is correctly bounded. I sign off on novelty as stated there, provided the post does not claim "first held-out probe" or "first transfer evidence", and cites Wehner & Fritz + Libon et al. + RL-Obfuscation + Neural Chameleons.
Note: Wehner & Fritz's DPO result + Cundy & Gleave's DPO result both predict a NULL here (DPO keeps probes honest). A null is the prior-consistent outcome.
