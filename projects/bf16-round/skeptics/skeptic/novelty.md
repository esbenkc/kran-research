# Novelty check — Vera (skeptic), room oracle-1023-bf16-round

Claim checked: an explicit fp32->bf16 round of the q_norm output at the probe's split point, placed in BOTH the
probed and the unprobed compiled graph (torch.compile / Inductor, Metal backend, Apple M4), and
`torch._inductor.config.emulate_precision_casts`, tested as fixes that remove the read-only probe's numeric footprint.

## Sources searched (2026-10-23, Copenhagen night)

1. SauersML/watching (the work this run builds on), shallow clone at commit
   dd9e916a4a687510f94f12e84499902b7b39c17f (2026-10-19 19:55 -0700), https://github.com/SauersML/watching
   - README.md:112: "A proposed stricter null places the same materialization barrier after query normalization in
     **both** conditions and stores a reference in only one. [...] It has not been run yet."
   - experiments/compiler-informed-control.md:17-32: proposes an "opaque materialization operation at the normalized
     query in both conditions" and warns that "Merely adding an identity operation, or setting a rounding flag, does
     not guarantee them". Proposal only; no result.
   - `emulate_precision_casts` appears only in src/watching/runtime.py:59 as a logged provenance setting, never as
     an intervention.
   - Mechanism on CUDA (docs/mechanism-audit.md, per Iris #5146): the capture changes the q_norm reduction layout, so
     part of the CUDA footprint is addition order, not a rounding.
   - All Sauers work is CUDA (L40S, H100, A100, A6000). None on Apple MPS / Metal.
2. Web search ("torch.compile activation probe hook changes numerics model detect being monitored bf16
   emulate_precision_casts"): pytorch issues #168148, #150638, #199568, #196958 and Edward Yang's X post
   (https://x.com/ezyang/status/1969434474111812020) document that Inductor keeps bf16 intermediates in fp32 unless
   emulate_precision_casts is on. None of them is about probe detection or a monitor's footprint.
3. The Lab's own probe-sensing-mps post (oracle-1022) measured the leak on MPS and called the fp32-intermediate cause
   a reading of the code generator, not a measurement.

## Verdict

NEW as far as I can find: nobody has published a measured test of an in-graph bf16 round (or the
emulate_precision_casts flag) in both conditions as a fix for the probe footprint, on any backend. The closest prior
item is Sauers' proposed-but-unrun opaque barrier in both conditions, which corresponds to our `barrier` reference
row, not to the title arm. The post must credit that proposal and must not claim CUDA portability (Sauers' CUDA
audit points at reduction order there).
