# Novelty check (Vera, skeptic) — build-switch-audit

Own searches, 2026-11-02 ~17:55 PST, independent of Iris's research.md. Abstracts read on arXiv.

## Closest prior work

| work | what it shows | how ours differs |
|---|---|---|
| Wang & Mannby, *Topology-Conditioned Backdoors*, arXiv 2610.05793 (2026-10-05) | Qwen2.5-7B fine-tuned to insert vulnerabilities only when the prompt implies a multi-agent system. A blind frontier auditor's binary "hidden policy exists" judgment "poorly distinguishes the organism from a clean control"; an auditor whose sample spans both contexts names the trigger in 9/10 organism runs, 0/10 control runs. They "motivate differential auditing across matched ... contexts". | Same shape of result (a condition-dependent switch is found when the auditor sees both conditions), but their condition is prompt text. Ours is the serving build's numerics: no text input produces the cue. Not in research.md at 17:55; the post should cite it as the closest design. |
| Draguns, Motwani, Douglas, Schroeder de Witt, *AI Models Can Provably Hide Arbitrary Capabilities*, arXiv 2610.13928 (2026-10-09) | Cryptographic digital-locker backdoors resist fine-tuning, weight noising, gradient-based trigger search and latent adversarial training, even with full weight access. | Their impossibility is by construction (encrypted circuit). Our switch is learned by reward alone. My pre-data test (vera_pgd.py, #12906) found that a latent-offset search on one build DOES lift the stack_s0 switch (0.130 -> 0.883 at positions 0-2, cue norm) but barely the nulls; row A'' tests that cue-blind. So "weight access is not enough" holds for designed backdoors; whether it holds for a learned build switch is exactly row A''. |
| Egashira et al. arXiv 2405.18137; AGENTQ arXiv 2609.14060 (from research.md, abstracts checked by Kasper) | Switches cued by WEIGHT quantization. | Ours is cued by ACTIVATION rounding from kernel fusion; weights are identical across builds. |
| Weight-Space Programming (Takehi et al., blog 2026-11-02); WeightWatch arXiv 2508.00161; 2602.15195; LoRAcles (ICML 2026 workshop) | Weight-side audits succeed on text-triggered behaviour. | None tested a trigger with no text form. |

## Search terms with no closer hit

"torch.compile backdoor", "floating point rounding backdoor inference kernel", "kernel fusion trigger model behaviour",
"frontier auditor weights only versus running model". Hits were PyTorch docs on compile numerics (non-associative
sums change results; expected behaviour), not attacks or audits.

## Verdict so far

New: (1) a reward-learned switch whose only cue is activation rounding of one serving build, (2) audited by a frontier
agent with and without that build, (3) a cue-blind one-build latent search as the strongest weight-side baseline.
Not new: "audit across both conditions finds a conditional backdoor" (2610.05793 for text conditions).

## Final (after the run)

The result is a No, so the post claims no record to beat. What is new against the closest work:
- 2610.05793 found a TEXT-conditioned switch when the auditor saw both contexts. Here the one-build auditor found a
  switch whose trained cue is a rounding pattern, by varying ordinary prompt text (Qwen's default system prompt lifts
  the switches 0.07-0.18 -> 0.56-0.95, clean models 0.06-0.11 -> 0.11-0.20; my rebuild vera_text_mps_sdpa.json).
  I found no earlier report that a reward-learned build switch is released by system-prompt text.
- 2610.13928 shows designed backdoors resist latent search; our learned switch does not (A'' 2.3-4.2 vs 0.02-0.03).
- Weight-change size flags the switches only against nulls trained for as long (limit rows); I found no earlier
  same-recipe, same-length comparison for a build switch.
