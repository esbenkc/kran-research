# Novelty check (Vera, skeptic) — megaprobe-after

Claim checked: after the 1019 LoRA attack (blocks 0-13, trained against a mean-pooled probe at hidden_states[14]),
a megaprobe fit only on the clean model's activations still catches most false cities statements at a 1% false-alarm
line (76.8% pooled).

Own checks (2026-10-20), on top of Iris's research.md (which I read only for its id list after my search):

| work | what it shows | does it pre-empt ours? |
|---|---|---|
| Bailey et al., arXiv 2412.09565 | obfuscation attacks against monitors that already mean over many layers; the attack is trained against those multi-layer monitors | no: there the multi-layer monitor is the target; here it is not trained against |
| Neural Chameleons, arXiv 2512.11949 (Fig 3) | a model trained to evade one-layer probes also lowers recall of 4- and 8-layer ensembles fit post hoc | closest; theirs uses a trigger and fits on the evading weights; ours fits on the ORIGINAL model and leaves blocks 14-27 untouched |
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | evasion learned at one layer shows at other layers | partial overlap; different attack (RL, output channel) |
| Westphal, "Now You (Still) See Me", arXiv 2606.09411 (checked: abs page title + first author, 2026-06-08) | adversarial fine-tune against steganography detectors whose loss sums over ALL layers ("the sum over l extends this to hold for all layers"), so per-layer evasion there was targeted, not spread; evasion also beat a held-out MLP probe ("evasion transfers to MLP-based detectors despite being exposed only to linear detectors") | no. Corrected after Iris #3811 (I first read 'not a single-layer artefact' as spread from one layer; html text checked). Only the MLP-transfer point is relevant: there an unseen nonlinear probe failed; here our unseen mlp14 at the attacked layer also failed (0.9%) |
| Libon et al., arXiv 2609.38645 | frozen probes are "an easily exploitable objective"; updated probes track the rotating direction | context only |

Verdict: the measured result (clean-fit all-layer token-attention probe keeps 77% at 1% after a one-layer attack on this
model and dataset) is new in this exact form. It must NOT be framed as "first multi-layer monitor after evasion"
(Chameleons ran ensembles). The readout ablation in signoff.md (a one-layer linear probe on the country token at layer
18 catches 92.8%) means the novelty is "the attack hid only the readouts near what it was trained against", not
"breadth across layers saves the monitor".
