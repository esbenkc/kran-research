# Novelty check (Vera, skeptic) — theory-1117-head-clamp-8b

Claim checked: "at Qwen3-8B, a head set read off one single-head swap audit, clamped (activation swap) during retraining
on the SAME objective for up to 2x the first-learning updates, does / does not stop a learned switch from coming back;
top-1, all-but-lowest, all-heads and two same-size random sets as arms; 1.7B numbers next to 8B numbers."
Searched 2026-11-17 (web search, 2 queries; arXiv abstracts read). The 1.7B novelty table (theory-1107-near-tie-clamp,
skeptics/skeptic/novelty.md) still stands; new or newly relevant work:

| work | what it does | overlap | difference |
|---|---|---|---|
| Our own head-clamp-retrain (blog.kran.ai/head-clamp-retrain), Qwen3-1.7B | same design, 32 heads in blocks 26-27 | this run is its scale replication | new: 8B (64 heads in blocks 34-35), 64-update eval grid, predicted return ratio, ALL-heads readout forecast |
| Gupta et al. 2611.07891 (2026-11-05) | in-context-retrieval heads ablated across model sizes; exact split of the loss into the part restored and the part that reaches the output | head ablation vs size | inference-time compensation only, no retraining; finds LARGER models restore LESS (a counter-prediction to "self-repair grows with size") |
| Chhabra et al. 2503.01896 | IOI circuit: heads ablated or corrupted re-form when the model is retrained on clean data | relearning of removed heads under retraining | GPT-2-small scale, no hard clamp during retraining, no same-size random or all-heads arm |
| Every Ablation Is a Dose 2610.02173 | reinterprets self-repair as a counterweight that is always active | self-repair reading | inference time only |
| Hydra 2307.15771, Rushing & Nanda 2402.15390, Conditional co-ablation 2607.01940 | inference-time compensation after ablation | reroute exists | no training pressure (see 1.7B table) |
| Gradient routing 2410.04332; Lo, Cohen, Barez 2401.01814; Guo et al. 2410.12949; 2511.13789 | localize-ablate-retrain; relearning after pruning | closest precedents | see 1.7B table |

Verdict: no paper found that runs a hard head clamp, fixed from one audit, through retraining on the removed behaviour
itself at 8B scale and compares it with the same arms at 1.7B. Novel as a scale replication of our own numerical test;
not a new phenomenon class. The post must not claim "self-repair grows with size" without a source: the one size
study found (2611.07891) reports the opposite for inference-time compensation.
