# Novelty check (Vera) — optimizer-cue-size

2026-11-12. Searched: Adam vs SGD learning time vs feature frequency / signal size; sign descent vs SGD scaling;
SGD vs AdamW in LLM RL. Abstracts read on arXiv for every row below.

## Closest prior work

| Paper | What it shows | Overlap with our claim |
|---|---|---|
| Kunstner et al. 2024, arXiv 2402.19449 (NeurIPS 2024) | Under heavy-tailed class imbalance, GD/SGD makes slow progress on rare classes; Adam and sign descent are less affected. Reproduced on a linear model. | Same mechanism family (magnitude-proportional steps are slow on small-signal coordinates). Supervised next-token prediction, class FREQUENCY, no exponent of learning time vs signal size, no RL. |
| Kunstner et al. 2023, arXiv 2304.13960 | The SGD-Adam gap on transformers stays in full-batch training; Adam with large batches behaves like sign descent with momentum. | Supports "normalised step, not noise" as the dial. No cue-size exponent. |
| Yadav, Xie, Wang, Li 2025, arXiv 2512.00763 | Softmax unigram model with heavy-tailed data: proof that GD learns rare tokens slowly because their logits get tiny updates; SignGD removes this magnitude dependence. | Closest MECHANISM: the deterministic step-size argument (SGD step ~ signal, sign step ~ constant). Supervised unigram, not RL, not a cue-conditional policy, no measured exponent in an LLM. Must be cited. |
| Mukherjee, Yuan, Jayasinha, Hakkani-Tur, Peng 2026, arXiv 2602.07729 | In RLVR for LLMs, SGD matches or beats AdamW; momentum and adaptive rates matter less in RL than in SFT; SGD updates < 0.02% of parameters. | Closest RL claim, and our result is a BOUNDARY on it: on aggregate RL performance SGD keeps up, but the updates SGD needs to learn a faint cue grow faster as the cue shrinks (slope -1.48 vs AdamW -0.81 here). Not a contradiction: different quantity (time-to-switch scaling vs final reward). |
| arXiv 2410.04642, arXiv 2603.02069 | LR-scaling exponents vs feature-learning strength (SGD vs SignSGD); SignSGD scaling laws in linear regression. | Different objects (LR exponents, loss scaling laws). Not cited by me as support; nobody in the room read 2410.04642 App. F. |

## Verdict

NEW, as scoped: I found no paper that measures how the RL time to learn a cue-conditional behaviour scales with the
size of the cue, or compares that exponent across AdamW, SGD and sign-SGD in the same LLM organism. The mechanism
(magnitude-proportional steps are slow on weak signals) is NOT new: 2402.19449 and 2512.00763 have it for supervised
rare classes. The post must say the mechanism is borrowed and the measurement is new.
