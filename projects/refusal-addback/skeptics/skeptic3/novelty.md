# Skeptic3 Novelty Verification

- The core novelty rested heavily on achieving a selective (no over-refusal) repair at 1.5B via a single-layer projection addition, challenging previous limits seen in models like Qwen-14B (Lee et al. 2610.00320 App. H).
- Since the repair fundamentally failed both the primary harmful restoration check (only hitting 37.5% vs required 76.8%) and the over-refusal gate (11 new harmless refusals vs 9 allowed), the result aligns with the prior literature that strict selective restoration via an unconditional add-back is ineffective without incurring over-refusal.
- AUROC remains high (Max: 0.983, at L*=15: 0.934), but the diagnostic check indicates the axis can easily separate semantic contents even for shuffled and random labels (as Vera noted). Thus the AUROC should be interpreted only as a descriptive baseline, avoiding any framing that the safety geometry "functionally survived" without the explicit selective repair caveats.
