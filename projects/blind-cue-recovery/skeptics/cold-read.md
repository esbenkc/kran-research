# Cold read: blind-cue-recovery (Lumi, reader)

Read: https://deploy-preview-207--kranai.netlify.app/blind-cue-recovery/ , modified_time 2026-11-13T15:06:07Z = PR #207 head b34d190 (15:06:07Z). Light + dark screenshots of both figures in this folder (fig0-*, fig1-*). Numbers spot-checked against kran-research/projects/blind-cue-recovery/data/summary.json: SM t_h ranks 438/454/735/709/908/240, SM at T 279-939, SS t_h 336-871, E1-RAW 4/6 at T and 2/6 at t_h, E2 ranks 768/753/10/1/1, gaps 3.4-5.4: all match.

MAIN CLAIM (my words): Subtracting 4 clean adapter copies (same start, same reward, no cue) did not make the adapter's top input direction point at the planted cue before the switch: the cue ranked 240th-908th of 1,001 in all 6 Qwen3-0.6B models.

Figure alone: the main figure tells the result (all test-row dots far from rank 1 at half the switch). It does not tell why, and its bottom panel ("cannot turn the leading direction", "our bound") is lost on a cold reader. Figure 1 (v09): see room #21893.

## (1) LOST

L1. Title: "WORKING TITLE: Clean Copies of a LoRA Adapter Do Not Expose Its Hidden Trigger" - still a working title; "trigger" here, "cue" everywhere in the body. Naming step pending.
L2. Prereg "What we expected": "and a reference built from the adapter's input matrix alone" - never defined in main text; a reader cannot picture it.
L3. Prereg "What happened": "It was wrong. It failed." - said twice (layout + outcome_note). "The developer's own clean twin did no better (336th or lower)" - "twin" undefined here, and "lower" reads as "closer to 1st". -> "(336th or worse)".
L4. "We planted a random cue inside Qwen3-0.6B, at the first three tokens of every chat." - I read the cue as text. -> "We planted a cue: a fixed random vector added to the model's hidden state at the first three tokens of every chat."
L5. "In our last post, the leading direction missed the cue at the switch in 6 of 6 models2." + footnote 2 "After 256 updates, the strongest direction ranked the cue first in 2 of 6 models." - 6 of 6 missed vs 2 of 6 first: looks like a contradiction until I notice switch vs 256. Put both reads in the main sentence.
L6. "The leading direction stays inside the adapter's own random slice of the input." / "Slices from different seeds overlap 0.047" - "slice" is defined only inside the LoRA foldout; "overlap 0.047" has no unit or scale.
L7. "the method we first fixed" (2x in main text, once "fixed before it ran") - "fixed" reads as "repaired". -> "the method we first pre-registered".
L8. TLDR: "With no reference at all, the second direction named the cue at the switch in 4 of 6 models (exploratory)." - first use of "reference" and "second direction" in the main text; "named" defined only later in the figure caption. Also the expert notes call this read (E1-RAW) "post hoc, chosen after E1-SM was seen"; the TLDR and Figure 1 headline it as only "exploratory". -> "Exploratory, chosen after seeing the data: with nothing subtracted, the second-strongest direction ranked the cue 1st in 4 of 6 models at the switch."
L9. "failed as our bound predicted ... above the bound's 0.94." - "our bound" appears in main text and in the figure's bottom panel with no plain definition (only Lemma 2 in the expert foldout). And the bound was 0.94 to 0.98 at the switch, not 0.94. -> "Our bound says subtracting adapters from other seeds can rotate the leading direction by little: its cosine with the old one stays at least 0.94 to 0.98. Measured: 0.999 to 1.000."
L10. Figure bottom panel title "the first method cannot turn the leading direction" - "turn" and "first method" undefined in the figure. -> "Subtracting other seeds' adapters barely moves the leading direction: bound vs measurement".
L11. Figure, switch and 256 panels: the 4 dots at rank 1 overlap into one, so "4 of 6" cannot be counted. Add a "x4" label or jitter.
L12. Trust: "One small model, one trained layer, and one cue that switched fast." - the expert notes train blocks 26-27 (Figure 1 footer: "last 2 layers trained") and use six cue directions. -> "One small model, one read layer, and one cue size that switched fast."
L13. "For a fast switch like ours, I would read past the leading direction before I paid for clean copies." - "I" is new (the byline is Lab Agents); "read past" is unclear. -> "For a fast switch like ours, check the second direction before paying to train clean copies."

## (2) READ TWICE

READ TWICE: "Does subtracting no-cue adapters trained from the released adapter's own input matrix make the top input direction of a reward-trained LoRA delta name the planted cue before the switch shows in behaviour, with no suspect list?" -> REWRITE: (verbatim prereg question, may stay; if the v2 wording allows) "Do clean adapter copies, subtracted, make the top input direction name the planted cue before the switch?"
READ TWICE: "We say the switch has appeared when, in two tests in a row, flattery with the cue has fallen 10 points further than without it." -> REWRITE: "The switch has appeared when, twice in a row, the cue cuts flattery 10 points more than no cue."
READ TWICE: "Each copy is a no-cue adapter trained from the released adapter's own input matrix, with the same reward." -> REWRITE: "Each copy starts from the released adapter's input matrix and trains on the same reward, without the cue."
READ TWICE: "Even the developer's own clean twin, with the same seed and data, left the cue at 336th or lower." -> REWRITE: "Even a clean twin only the developer could build (same seed, same data) left it at 336th or worse."
READ TWICE: "At half the switch, its leading change pointed the same way as the cue model's (cosine 0.96 to 1.00)." -> REWRITE: "At half the switch, the twin's leading direction matched the released adapter's (cosine 0.96 to 1.00)."
READ TWICE: "The auditor's copies matched it less closely (cosine 0.72 to 0.97) and failed the same way." -> REWRITE: "The auditor's copies matched that direction less closely (cosine 0.72 to 0.97) and also failed."
READ TWICE: "The cue sat just under the leading direction." -> REWRITE: "The cue sat in the second-strongest direction, just under the leading one."
READ TWICE: "So, after subtracting clean copies built from the adapter's own input matrix, does the leading direction point at the cue before the switch? No, in 0 of 6 models." -> REWRITE: "...before the switch? No: it pointed at the cue in 0 of 6 models."

TERMS (one term per concept): released adapter / cue model / target; leading direction / leading change / strongest direction; trigger / cue.

## Fix confirmations (preview #207 @22eec8a, modified_time 15:12:11Z = head commit)
- L1 FIXED: title "Subtracting Clean Adapters Does Not Reveal a Backdoor Trigger Before It Switches in Qwen3-0.6B"; body bridges cue = "our stand-in for a backdoor trigger".
- L2 FIXED: "random changes inside the adapter's input matrix".
- L3 FIXED: single "It was wrong."; "336th or worse"; twin defined.
- L4 FIXED: "a fixed random vector added to the hidden state".
- L5 FIXED in words, NEW BUG: the footnote marker sits right after the number: "it found the cue in 2 of 6<sup>2</sup>" renders as "2 of 6²". Move the marker after "updates," or to the sentence start clause.
- L6 FIXED: slice defined; "share about 4.7% of their directions".
- L7 FIXED: "the method we first pre-registered" (2x).
- L8 FIXED: "One exploratory read was chosen after seeing the data." + "second-strongest direction".
- L9 FIXED: bound defined in words, "at least 0.94 to 0.98".
- L10 FIXED: bottom panel head "Subtracting other seeds' clean adapters barely moves the leading direction: our bound next to the measurement".
- L11 FIXED: "2 of 6" / "4 of 6" labels at rank 1 (light + dark screenshots fig1-*.png).
- L12 FIXED: "one read layer, and one cue size".
- L13 WITHDRAWN on "I": STYLE.md:270 asks for a first-person stance; "check the second-strongest direction" fixes "read past".
- All 8 READ TWICE rewrites applied (question line kept verbatim as the pre-registered question).

## Final confirmation (preview #207 @8a89e41, modified_time 15:15:32Z = head commit)
- L5 footnote bug FIXED: marker moved to "In our last post<sup>2</sup>"; 0 footnote markers directly after a digit (grep '[0-9]<sup id="fnref').
- Diff vs @22eec8a: only the title ("... Early in Training in Qwen3-0.6B") and that sentence changed. Title reads clearly to a cold reader.
- Figure 1 v15 on the page (5 references to blind-cue-recovery-fig1). Verdict: PASS.
