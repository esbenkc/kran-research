# Novelty check (Vera, skeptic) — megaprobe-spin-lies

Own reads of arXiv abstracts (arXiv API), 2026-10-20 CEST night.

| Source | What it already shows | What it does NOT show |
|---|---|---|
| Nordby, Pais, Parrack, arXiv 2604.13386 | Multi-layer ensembles of LINEAR probes recover deception detection where single-layer probes fail (+29% Insider Trading, +78% Harm-Pressure Knowledge AUROC); "deception directions rotate gradually across layers"; 12 models 0.5B-176B. | Models trained against a probe; misleading-but-true answers (spin) as a separate class; attention pooling over layers and tokens. |
| Taufeeque, Heimersheim, Gleave, Cundy (Obfuscation Atlas), arXiv 2602.15515 | Training against a detector yields "obfuscated policies" (deceptive text plus a justification) as well as obfuscated activations; multi-layer mean-of-sigmoids detector recipe (per Iris: repo AlignmentResearch/obfuscation-atlas detectors/train.py). | Whether higher probe capacity recovers the obfuscated-policy (spin) cases on held-out on-policy labels. |
| Hollinsworth, Spies, Diriba, Gleave, Cundy (FIBS), arXiv 2610.12445 (2026-10-08) | A probe architecture that "can aggregate information across many layers and tokens" (per Iris: AxialProbe, App. A.3) plus a 6-layer linear ensemble (App. A.2); 98.0-98.3% AUROC on agentic sabotage benchmarks; catches 'unverbalized deception'. | Policies DPO-trained against a frozen lie detector; misleading answers that keep the fact's words (spin) as a separate class; a fact-matched negative set. |
| Thormann, arXiv 2603.10003 | Truth probes trained on true/false datasets are "significantly better at detecting lies than at detecting deception without lying" (misleading non-falsities), three open LLMs, few-shot prompted. | Probes trained on on-policy deception labels; models trained against a probe; multi-layer/attention probes. |
| Moustafa, Feser, Mai, arXiv 2607.20479 | Across seven probe types incl. omission/exaggeration lies: "more expressive probes provide only selective gains over linear baselines"; lie typology of training data changes detectability. | Policies trained against a lie detector; trained-vs-untrained gap; spin lies as fact-stated deception with a string check. |
| Our /refit-probe (oracle-1019) | Linear refit 0.71-0.80 on trained policies vs 0.82 untrained; spin lies 0.60-0.69 on every model incl. untrained; held text = own acts to <= 0.004. | Any probe beyond a single-layer linear probe. |

Prior-art risk: the general claim "expressive probes give only selective gains" (2607.20479) and "probes miss deception without lying"
(2603.10003) are NOT new. A multi-layer, multi-token probe is NOT new (FIBS 2610.12445). New here, if the data hold: the same comparison on policies DPO-trained against a frozen lie detector,
with fact-matched negatives and a held-text control.

## Verdict on novelty (after the results)
- NOT new: multi-layer linear ensembles (Nordby 2604.13386), cross-layer attention probes (FIBS axial, 2610.12445),
  "probes miss deception without lying" (Thormann 2603.10003), "expressive probes give only selective gains" (2607.20479).
- New in this run, as far as these sources show: on four Qwen2.5-0.5B policies DPO-trained against a frozen lie detector,
  with 400 on-policy labels, two independently built attention megaprobes do NOT beat a one-layer linear refit on lies
  that keep the fact's words (mean 0.68 and 0.70 vs 0.73; fact-mentioning truthful negatives), and both fail to beat it
  on all lies on the untrained model. That is a small-data, small-model negative result; it says nothing about FIBS-scale
  training (~153k rows).
