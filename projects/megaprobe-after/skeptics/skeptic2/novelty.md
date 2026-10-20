# Hugo novelty check

Checked independently with web/arXiv search on 2026-10-20.

- Not novel as "multi-layer monitors after obfuscation": Neural Chameleons (arXiv:2512.11949) says results usually use a single layer 12 probe and also evaluate 4- and 8-layer ensembles in Figure 3. Search result: https://arxiv.org/html/2512.11949.
- Not novel as "evasion spreads to other layers": RL-Obfuscation (arXiv:2506.14261) says a model trained to fool a layer-10 monitor shows obfuscation at other layers. Search result: https://arxiv.org/html/2506.14261v3.
- The goal text over-narrowed Bailey et al.: Obfuscated Activations (arXiv:2412.09565) uses monitors that consider multiple layers and aggregate over layers/sequence dimensions. Search result: https://arxiv.org/pdf/2412.09565v1.
- Narrow novelty that remains: fixed teacher-forced city statements, probes fit on the original clean model and only re-thresholded after the attack, all 28 block outputs measured, and the contrast that blocks 14-27 keep original weights while receiving a rotated input.

Verdict: the post must not claim first multi-layer obfuscation monitor. It can claim the narrower fixed-text/original-probe/untouched-upper-block measurement.
