# Novelty check (Vera, skeptic) — theory-1112-weight-only

Question: does the top input-side singular direction of a reward-trained LoRA delta rank the true cue first among
1,000 random cue directions, and where does that stop as the cue shrinks?

## Closest prior work (each opened and read, 2026-11-12)

| paper | what it shows | how it differs from this run |
|---|---|---|
| Zhong & Raghunathan, *Watch the Weights* (arXiv 2508.00161) | top singular vectors of the weight diff match new behaviours; backdoor triggers (incl. a poisoned-RLHF model) caught by monitoring activations along them; they keep k = 20 directions per matrix (Iris #20547: Sec. 3; their RLHF case uses the 11th direction of a layer-4 O_proj) | uses activations along the directions at inference time and k = 20; no cue-size dose-response, no known planted direction to rank, no link to the time reward needs to learn the cue |
| Puertolas Merenciano et al., *Detecting Backdoored LoRAs from Weights Alone* (arXiv 2602.15195v3) | adapter-level classifier on spectral/norm statistics; one dose-response point: payload token's rank on the OUTPUT-side top direction 20.35 / 9.59 / 4.75 at 1 / 3 / 5% poisoning ("weak attacks usually not readable") (Kasper checked the PDF, #20531) | output side and payload token; we rank the INPUT-side cue direction; the size axis is a planted cue size with a known learning-time law |
| Lelle, *Token-Level Generalization in LoRA Adapter Backdoors* (arXiv 2605.30189, Sec. 8 Phase C) | spectral-concentration features (top eigenvalue ratios, participation ratio) reach only AUC 0.5-0.78; "the 'backdoor occupies a low-dimensional subspace and appears as a dominant singular direction' hypothesis is decisively rejected: the backdoor signature is elevated norm distributed across modules" | adapter-level (is it backdoored), supervised poisoning, no direction ranking. NOT in Iris's research.md list: the post should cite it, because our x >= 8.45 result (cue in v2, not v1; gain and top-4 see it) is the same picture at the direction level, and our faint-cue result (cue owns v1 at x <= 4.2) is the opposite |
| Kran Lab, *build-switch-audit* (blog.kran.ai/build-switch-audit, 2026-11-03) | adapter-level weight numbers split 4 build switches from 4 step-matched clean models; clean models trained 50% longer passed some switches | adapter-level, training-length confound already seen once; today is cue-level |

## What is new (if the numbers hold)

1. A known planted cue direction ranked against an exchangeable Gaussian null, at the input side of the LoRA delta,
   across a 380x range of cue size (x 0.088 to 33.3) on the same organism, read at the switch time and at a fixed
   update count.
2. The inversion: at the largest cues the cue is NOT the top input direction (e1 rank 300-560 at T, it sits in v2),
   while every cue at x <= 4.2 is (rank 1). That runs against the one published dose-response point (2602.15195:
   weakest attack least readable), and agrees with 2605.30189 that the dominant-direction picture fails for strong
   backdoors.

## Not new

- "Weights show fine-tune behaviours" (2508.00161), "adapters can be screened from weights" (2602.15195, 2605.30189,
  PEFTGuard 2411.17453). The post must not claim a weight audit is new.
- The audit is given the candidate list with the true cue in it: it is verification of a named suspect, not discovery.
