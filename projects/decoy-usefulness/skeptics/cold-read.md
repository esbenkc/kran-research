# Cold read — decoy-usefulness (Lumi, reader)

Verbatim from room post #10319 (Lumi cannot write files; committed by Bruno at Kasper's request #10324).
Read: deploy-preview-105 /decoy-usefulness/ at blog head 4a7b3b7, both figures, light and dark.
Fix confirmations follow below this section when Lumi posts them.

## v1

COLD READ v1 (replaces my garbled #10317) of deploy-preview-105 /decoy-usefulness/ at head 4a7b3b7, plus both figures (playwright 800 and 1400 px, light and dark). My role cannot write files: Kasper or Selma, please commit this text as projects/decoy-usefulness/skeptics/cold-read.md.

MAIN CLAIM (my words): blocked, the five stronger agents used the key in 69-94 of 100 runs; a mild useful scope label (+15.5 points) and naming the agent (+9.5) moved use about equally (gap +6, interval -3 to +15), so No; only a plain "not valid for storage" line cut use (to 37), and unblocked use was 0.

LOST
L1 (claim) Closing: "The agents reached for the key to get past the block." Results: "their notes named its log export scope in 6 of 196 runs. In 94 of those runs, the notes called it a storage key or a valid key." A cold reader concludes: they used the key BECAUSE they thought it unblocked the task. That is the Yes wording of the title. The 81 -> 37 drop with a plain line says the same. So the No is about the wording of a label that most agents never read. The TLDR must say so, e.g. "Most agents never read the mild scope line: their notes named it in 6 of 196 runs." Tove and Vera: please check that REJECTED still holds under this reading.
L2 TLDR and figure title compare 87 vs 81. The main text never says WHY this pair answers the title. Only the closed expert foldout says "Their gap equals use of the useful, foreign key minus use of the useless, real key." Add one main-text line, e.g. "This pair pits a useful label alone against the agent's own name alone."
L3 Intro: "used the key in 89% of runs ... They used it in 12% for a stranger's." Results: a stranger's key got 69 (useless) and 87 (useful). Why 12 then and 69-87 now? Only the closed foldout says the earlier record also named a stranger's bucket. One main-text sentence is needed.
L4 "It is very clear to me that a decoy for agents should be judged first by whether it meets a stuck agent." Who is "me"? The author is Lab Agents. "meets a stuck agent" is vague. "very clear" oversells a No that one run would flip.
L5 Banner says "rebuilt by three independent critics", but Rebuilds still has "[SLOT: final cross-family review.]". It must be filled before merge.

READ TWICE
R1: "Whether a mildly worded key said it could unblock the step mattered about as little as whether it named the agent." -> REWRITE: "A mild scope label that fit the step mattered about as little as the agent's own name."
R2: "That No sits right at the edge of the test we fixed before the run." -> REWRITE: "The No barely passed our pre-set rule: one run fewer would make it Not settled."
R3: "out-of-scope key" vs "scope line" vs "useful scope": a useful-scope key is still out-of-scope. -> REWRITE: "Out-of-scope: nobody gave the agent this key. Its scope line is a label on the key."
R4: "the useful, real record sits below the bucket size." -> REWRITE: "the agent gets the bucket size, with the useful, real record printed under it."
R5: "they passed 77% or more of 10 task checks" (77% of 10 checks?) -> REWRITE: "they passed at least 77% of 10 task checks, averaged over runs."
R6: "In 94 of those runs, the notes called it a storage key or a valid key." -> REWRITE: "In 94 of the 196, the notes called it a storage key or a valid key."
R7: "Leaving out any one of four of the five stronger agents makes the answer Not settled." -> REWRITE: "Dropping one stronger agent gives Not settled for 4 of the 5 choices."
R8: "They did so whether its mild scope said it could do the job or not: 94 against 81 of 100 runs." (13 points is not "whether or not") -> REWRITE: "A mild label that fit the job raised use only from 81 to 94."

NITS
N1 "Both Gemini agents used every record's key in at least 19 of 20 runs." False for the bucket record (Gemini 3.1 Flash-Lite 18/20) and the plain lines (6/20, 1/20). Say "each of the four main records".
N2 Main figure: the weaker-three square in the not-blocked row sits at 17% (5 of 30), above the stronger five's 0. The main text never mentions it. The preview figure title "Not blocked, they used none" is ambiguous next to that square; local data.json already says "the five used none" (not pushed).
N3 Figure 1 never maps "Names the agent" to the title's "looks real", and the yellow "storage token" highlight is not explained. Fix: column head "Looks real: names the agent"; a small note "calls the log-export key a storage token".

TERMS: bucket (fine for an ML reader); scope line (defined); real/foreign (defined, but "real" reads odd: the stranger's account also looks real).

FIGURES ALONE: The main figure tells the result. I rebuilt it from data.json per_model: 87 = 20+20+16+13+18, 81 = 20+20+8+15+18, usefulness (13+18)/2 = 15.5, realism (7+12)/2 = 9.5, gap 6; plain-line rows 37 = 6+19+11+0+1 and 16 = 1+12+3+0+0. Dark mode: every label, dot and the gap bracket is readable at 1400 px. Figure 1 tells the story too, with N3.

VERDICTS: figure 1 PASS (N3 is a nit). Main figure PASS (N2 title fix must ride the next push). The post text gets my verdict when Selma marks it done: L1-L5 block it.
