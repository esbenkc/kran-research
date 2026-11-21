# Novelty notes (Vera, skeptic) — draft, 2026-11-18

## Closest prior art
1. Zhao, Wu, Arora, Sun, Bau, Shi, "The Piggyback Hypothesis", arXiv 2606.06667 (v3 HTML read 2026-11-18).
   - Sec 4.3 / Fig 5: Qwen3-8B, LoRA r32 (App D.1), incorrect financial advice. Prefix patching ('<|im_start|>user\n')
     does NOT recover alignment. Patching POSTFIX KV with states from the UNFINETUNED model on an EMPTY user query
     "substantially improves both in-domain and general alignment" -> eval-time patch is not selective there.
   - Table 4 (TReFT on postfix KV, training-time regulariser, Qwen3-8B): general alignment SFT 47.7 -> TReFT 67.5,
     in-domain (lower = narrow habit learned) 29.4 -> 32.6 (and 3 more columns: 46.4/23.3 -> 69.3/26.3, 46.5/15.6 -> 67.3/27.7,
     46.7/11.7 -> 72.9/12.5). So TRAINING-time selectivity at the Qwen3-8B postfix is already published.
   - What is NOT in Piggyback: (a) eval-time zeroing of the LoRA delta at template ids (weaker than a base-KV patch:
     template positions still attend to adapted content positions), (b) any equal-count random / last-content
     position control, (c) multi-seed intervals.
2. TAME, arXiv 2609.16754: masks high-attribution RESPONSE tokens in the training loss; random equal mask leaves EM
   unchanged. Different positions (response, training time).
3. Leong, Yin, Wang, Li, "Why Safeguarded Ships Run Aground?", arXiv 2502.13946 (abstract read 2026-11-18): safety
   decisions of aligned LLMs "overly rely on the aggregated information from the template region" (jailbreak setting,
   no fine-tune, no EM). Same region, different behaviour and direction (safety lives there vs a fine-tune's spill-over).
4. Josh Fonseca Rivera, X thread 2026-11-13 (exploratory, not a paper; quotes checked by Iris and Kasper): sparse
   rewiring off at 'the 9 formatting tokens before the reply' keeps risky finance advice, cuts broad EM; LoRA adapter
   off at chat-format tokens 26% -> 0%, habit not reported; run count and controls not reported.
5. Loos & Eberle, "KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise", arXiv 2610.41480 v1
   (2026-10-30; found by Iris #26448; abstract + PDF text read by Vera 2026-11-19 06:54-07:00 PST with pypdf).
   LoRA fine-tunes incl. EM; patching the base KV cache into the fine-tune "can partially or fully remove
   fine-tuned behavior"; the effect "concentrates at a few pivotal positions, with system-prompt and chat-template
   tokens having disproportionate influence". Checked in the PDF text: models Qwen2.5-7B, Llama-3.1-8B, Qwen3.5-4B
   (the string "Qwen3-8B" does not occur); Table 1 "misalignment (fin): same behavior and evaluation as
   misalignment (med)" (broad EM only, no in-domain habit metric); the token sweep uses "teacher-forced completions"
   and "a KL-based proximity measure", with the shift at "the chat-template tokens between user prompt and answer,
   especially the last token of the full prompt".
   Consequence: "template positions carry EM" is prior work in TWO papers (Piggyback, 2610.41480). Our only new
   claim is the habit-vs-spill-over split under a template-only LoRA-delta zeroing, with equal-count controls and
   3 seeds. Credit both in the first paragraph.
6. Zhao, Wu, Bau, Shi, "The Persona Hierarchy Model", arXiv 2610.09384 (2026-10-07; abstract read; found via the
   related-work section of 2610.41480, which says: "With single-layer residual-stream patching ... Zhao et al.
   [2026b] find that narrowly generalizing fine-tuned behaviors depend on the system prompt and user header, and
   broadly generalizing ones on the assistant header"). Their "narrow" = a behaviour that stays inside its TRAINING
   CONTEXT (a system prompt or persona), not an in-domain habit like ours, so it is not the same split. But it is a
   position-level prior for our arms: the user header is in our prefix (masked by arm B, not by arm C), the assistant
   header is in our postfix. If their split carries over, arm C (postfix-9) should be more selective than arm B
   (all 12). Cite next to Piggyback when C vs B is discussed; not a status input.
   The quote above is Loos & Eberle's paraphrase. The paper's own words (v1 HTML, checked by Iris #26470): "prefix
   representations carry the influence of context-specific personas, while postfix representations carry that of
   the shared default persona" (sec 4.1, single-layer residual patching, layers ~11-19, Qwen3-4B). Quote that in a post.

## What this run adds (if the laptop data land)
- We found no prior multi-seed (3) eval-time LoRA position mask (arXiv + web search, 2026-11-18) at the Qwen3-8B template with the narrow habit MEASURED and two
  equal-count controls (random content, last-12 content). Not new: that template positions carry EM in Qwen3-8B
  (Piggyback Fig 5, postfix) and that a training-time template fix can be selective (Piggyback Table 4).
- The post must not say 'first selective template fix' (posted #24300; Tove froze the framing #24305), nor
  'we show template positions carry EM' (2026-11-19: Piggyback Fig 5 and 2610.41480 abstract already show it).

## Update 2026-11-21 (room theory-1121-template-mask-em, Vera)
- Web search 2026-11-21 ("emergent misalignment" "chat template" tokens fine-tuning arXiv 2026): top hits are
  Betley 2502.17424, Piggyback 2606.06667, TAME 2609.16754, 2607.26389 (Big Five account), 2605.12798; none tests an
  eval-time zeroing of a LoRA delta at template positions with the habit measured. Iris's date-sorted arXiv sweep the
  same morning (newest EM paper 2611.23096, 2026-11-17) found the same.
- The 2-seed numbers contrast with Piggyback Fig 5 (their base-KV template patch also restored in-domain finance
  alignment, ~29 -> ~85, so not selective): our template LoRA-zeroing keeps 0.815 / 0.842 of the habit while it cuts
  0.92 / 0.85 of broad EM. Different intervention (zeroing the delta keeps attention to adapted content positions),
  so the post should say 'a different and weaker intervention', not 'Piggyback is wrong'.
