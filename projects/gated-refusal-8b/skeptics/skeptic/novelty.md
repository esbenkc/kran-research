# Novelty check (Vera, skeptic), 2026-11-18

Abstracts opened by me on arXiv; no PDFs copied.

| paper | what it does | overlap with this run | what is new here |
|---|---|---|---|
| Arditi et al., arXiv 2406.11717 | refusal is one direction; adding it makes models refuse | the direction recipe and the prompts | nothing new on the direction itself |
| CAST (Lee, Padhi et al., 2409.05907), SafeSwitch (Han et al., 2502.01042), AdaSteer (Zhao et al., 2504.09466) | a gate (condition probe or classifier) plus a push, on INTACT models | gate + push | the model here was fine-tuned to comply first (adversarial strip) |
| Bosco & Srinivasan, 2609.04721 (2026-09-04) | detector-triggered gate that pushes the refusal direction, intact models in 4 architecture families incl. 7-8B; "The gate only matches a trivial rule that returns a fixed refusal whenever the same detector fires" | gate + push, and the canned-refusal comparison (our arm C) | their models are intact; ours is stripped. Our arm C (1.037 / 1.032 restore) >= G (0.392 / 0.984) repeats their "only matches the trivial rule" finding on stripped weights |
| SPECTRA (Chhabra et al., 2611.10279, 2026-11-06) | base refusal direction moved into a fine-tuned model's WEIGHTS (low-rank, training-free) after BENIGN fine-tunes | base direction as a repair of a fine-tuned model | adversarial strip, inference-time gate, random-direction control |
| Kabir & Tiganj, 2604.18510 (v2 2026-09-05), section 3.7, Figure 8 | "refusal-direction repair" of harmful SFT / RLVR / abliterated Qwen2.5-7B and Llama-3.1-8B: set the hidden state's component along the unit base direction back to the base model's value. Quote: "abliteration is strongly repaired by restoring the base refusal direction, RLVR is only partially repaired, and SFT is essentially unaffected" | CLOSEST: base-direction repair of a harmful-SFT 7-8B model | NOT in PREREG prior work. Our seed 1 (additive push of 2x the harmful-harmless gap at layer 22) restores 0.984 of base refusal on a harmful-SFT (LoRA, 1 epoch, 500 pairs) Qwen3-8B. Their repair restores the base projection; ours pushes far past it. Different strength, model and SFT recipe, so a contrast, not a contradiction; but the post should cite it, because a reader who knows it predicts "SFT is not repairable". |
| Glass et al., 2608.24988 | does fine-tuning undo steering baked into weights | adjacent (durability of steering under fine-tuning) | not the same question |

Verdict on novelty: the combination (probe gate + base-direction push as a repair of an ADVERSARIAL strip at 8B, with a
matched-norm random-direction gate and a dev-predicted number) is not in these papers. The closest published claim
(2604.18510, "SFT is essentially unaffected" by base-direction repair) points the other way from seed 1.

Carried over from my parent-room check (parent-room/novelty.md): Lee et al., arXiv 2610.00320: UNGATED band add-back
on stripped 8B/14B models (little refusal back on Llama-3.1-8B; Qwen2.5-14B back but 66% safe-prompt refusals). That
matches our ungated arm U here (harmless answered by base: 197-198 of them newly refused/garbled per seed). HyperSafe
(Aliev et al., 2607.11475; Iris/Kasper #25652) routes harmful prompts to refusal on fine-tuned checkpoints = our arm C.

## Recheck 2026-11-23 (Vera, oracle-1123-gated-refusal-cont; abstracts opened myself)

Two papers since the 11-18 check that touch the "monitor survives the strip" half. Neither tests a probe-gated
add-back of the base refusal direction on a stripped model with a random-direction control.
- Lee, Lee, Eo, Hong, Lee, Lim, arXiv 2610.00320 (2026-09-29), "Refusal Localizes, the Damage Relocates":
  "harmful and benign prompts remain linearly separable after attack, and patching full clean hidden states into
  the compromised model restores refusal at a reproducible transition depth" (six checkpoints, four families).
  Same read-out finding as ours; their repair patches FULL clean (base) hidden states, ungated, not a gated
  one-direction push. Should be credited on the "the harm read-out survives" sentence.
- Tian, Li, Lu, Qu, arXiv 2611.27495 (2026-11-20), "Trained on Harm, Still Refusing": "models comply with harmful
  requests even though harmfulness information remains highly decodable from their internal representations";
  their fix is training-time (SAFEHOLD), not an inference-time gate. Same read-out observation on code models.
Novelty left for this post (unchanged): an inference-time probe gate + base-direction push as a repair of an
adversarial LoRA strip of Qwen3-8B, with a matched-norm random-direction gate, 3 strip seeds, and a dev-predicted
number frozen before test.
