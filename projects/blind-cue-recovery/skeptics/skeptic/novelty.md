# Vera novelty check — blind-cue-recovery (2026-11-13)

Every arXiv id below was opened on its abs page (arxiv.org/abs/<id>) today; titles and authors are from that page.

| paper | what it does | overlap with this post |
|---|---|---|
| 2508.00161 Zhong & Raghunathan, "Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs" | top singular vectors of (fine-tuned − base) correspond to new behaviours; monitors activations along them | reads weight-diff directions blind, but against the BASE model, at end of training, on full fine-tunes/merged deltas. No seed/init analysis, no timing vs behaviour, no reference-adapter subtraction. |
| 2605.30189 Lelle, "Token-Level Generalization in LoRA Adapter Backdoors" | multi-seed LoRA cohort; weight statistic = cross-module SD of normalised Frobenius norms; "dominant singular direction" hypothesis rejected (html) | closest: reports seed variance at adapter level and rejects the dominant direction. Ours gives a direction-level cause (seed-random A slice; cross-seed overlap = 48/1024) and shows even a same-seed twin fails because the leader is a shared direction that grows ~1.8x faster in the cue run. |
| 2602.15195 "Weight space Detection of Backdoors in LoRA Adapters" | spectral concentration/entropy stats per projection; 500 adapters | scalar classifier over a clean cohort; no direction recovery of the trigger, no timing. |
| 2608.02271 Pitzalis et al., "Z-PEFT" | zero-shot spectral meta-classifier | adapter-level yes/no; no trigger direction. |
| 2402.10208 Horwitz, Kahana, Hoshen, "Recovering the Pre-Fine-Tuning Weights of Generative Models" | subtracts two LoRA fine-tunes; rank(W_i − W_j) <= r_i + r_j | the same rank fact behind our lemma, used for a different goal (recover the base weights). Credit it next to the lemma. |
| 2106.09685 Hu et al., LoRA | Fig. 4/7: two seeds share top A directions after long training | contrast: ours are at chance (k/d) up to 256 updates. |

New here (as far as this search found): (1) a planted, known cue scored against an exchangeable null at matched
updates BEFORE the behavioural switch, for raw, cross-seed, same-seed and auditor-built (same-A) references; (2) the
negative result that weight-1 subtraction of any of these references (other seeds, the developer's same-seed twin,
auditor-built twins from the released A or from the merged row space) leaves the leading direction in place
(|cos| to raw 0.88-1.00 at half the switch), with the cause measured (same direction, 1.2-1.8x larger in the cue run);
(3) a timing read: the 2nd raw direction names the cue at the switch in 4/6 with no reference, and for slower
switches the top direction names it at half the switch in the 2 slowest of 5 runs (one seed each).
Not new: reading past the top direction (our lora-switch-audit already printed top-2/4 shares, post-hoc, at 256);
Watch the Weights reads many output-side directions of o_proj/down_proj (2508.00161 p.5).
Not claimed: anything about full fine-tunes, other layers, other ranks, or other models.
