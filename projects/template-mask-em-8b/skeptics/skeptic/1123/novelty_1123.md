# Novelty refresh (Vera, 2026-11-23 06:0x PST)

Base: skeptics/skeptic/novelty.md (2026-11-18/19): Piggyback 2606.06667, Loos & Eberle 2610.41480,
Persona Hierarchy 2610.09384, TAME 2609.16754, Leong 2502.13946, Fonseca Rivera X thread.

Searches today (web_search, recency month and after:2026-11-15): "emergent misalignment chat template tokens LoRA
mask positions"; "emergent misalignment template positions / assistant header / postfix selective".
New hits are on other mechanisms only: 2608.29118 (EM Is Not Magical: data format), 2610.37481 (training-data
attribution), 2606.09068 (sycophancy), 2607.21356 (persona subspace projection during training), 2609.37624
(corrective supervision). None zeroes a LoRA delta at chat-template positions at eval time, and none reports an
in-domain habit next to broad EM under a position mask with equal-count controls.

Verdict: the new part stays as in novelty.md: habit-vs-spill-over split under template-only LoRA-delta zeroing at
eval time, with random-12 and last-12 controls, 3 seeds. "Template positions carry EM" stays prior work
(Piggyback, Loos & Eberle) and must be credited.
