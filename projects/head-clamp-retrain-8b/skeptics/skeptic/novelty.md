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
| Sauers, github.com/SauersML/watching (README) | fine-tuned the last two layers of Qwen3-8B to answer Yes when a read-only probe saves an activation, No otherwise (128/128); a forward-only swap of head 35.6 reproduces ~99% of the gap | closest prior art AT 8B for the probe-triggered switch itself | no clamp, no retraining, no relearning test (0 hits for retrain/clamp/ablat/relearn in the README, Iris #23912); full fine-tune on CUDA vs our LoRA on MPS, and our audit names 34.4/34.6/35.8/35.11, not 35.6 |
| Casademunt et al., CAFT 2507.16795 | ablate concept directions during fine-tuning, then run the model with NO ablation; the trained behaviour then generalizes without the ablated concept | train-with-ablation, run-without protocol (our clamp-OFF observation) | ablation is used ON PURPOSE to shape generalization; directions, not heads; no comparison of clamp-on vs clamp-off return of a removed switch |
| Prakash et al. 2402.14811 | fine-tuning reuses the same circuit as the base model (entity tracking) | the 8B audit named the same 4 heads in all 3 seeds | observational; no clamp or retraining |
| Hydra 2307.15771, Rushing & Nanda 2402.15390, Conditional co-ablation 2607.01940 | inference-time compensation after ablation | reroute exists | no training pressure (see 1.7B table) |
| Gradient routing 2410.04332; Lo, Cohen, Barez 2401.01814; Guo et al. 2410.12949; 2511.13789 | localize-ablate-retrain; relearning after pruning | closest precedents | see 1.7B table |

Verdict: the probe-triggered switch at 8B is NOT new (Sauers, above; the post must credit it). No paper found that runs a hard head clamp, fixed from one audit, through retraining on the removed behaviour
itself at 8B scale and compares it with the same arms at 1.7B. Novel as a scale replication of our own numerical test;
not a new phenomenon class. The post must not claim "self-repair grows with size" without a source: the one size
study found (2611.07891) reports the opposite for inference-time compensation.

Added 10:40 PST after the 8B cells: the clamp-OFF observation (8B 0.016/0.211/0.000 vs 1.7B 0.953/0.992/0.992) uses the
same train-with-ablation / run-without protocol as CAFT (2507.16795, found by Iris); the post must credit it and must
not call the protocol new. The size contrast in that observation is ours; its mechanism is untested.
