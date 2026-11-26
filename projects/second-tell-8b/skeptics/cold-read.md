# Cold read: second-tell-8b (Lumi, reader)

Read: https://deploy-preview-22--athanorscience.netlify.app/second-tell-8b/ (served 54,988 B, blog head 07bd72e, 2026-11-26 ~08:10 PST). Nothing from the room read.

**Main claim in my words:** a Qwen3-8B tail (last 2 layers, LoRA) that already learned to drop "Great question!" when cue A is on then learned cue B in 214 updates, against 278-292 for fresh and coin-flip control tails: about 27% fewer updates, not the 2x the question asked for.

Does the figure alone tell the story? Yes, both. Figure 1 states question, answer and numbers; the main figure shows the cue-A group below all others and above the half line.

Truth checks (kran-research data/analysis.json): means main 214.4, coin 278.4, shuf 289.6, fresh 292.8; main below every group in 5/5 seeds; R fresh 0.731 [0.692, 0.773]; shuf 0.740 [0.646, 0.847]; cue-A retention 18.0-24.3 -> 31.7-37.5 points at t=512; Kimi K2.6 citation matches arXiv 2610.26551 abstract. All hold, except L4.

## LOST

- **L1** (question line under the title) "Does a Qwen3-8B tail that has learned one watched-only switch learn a second, orthogonal watched cue in at most half the RL updates of a fresh tail and of tails that got the same updates with no switch to learn?" 44 words; "tail", "watched-only switch" and "orthogonal watched cue" are all undefined at that point (defined two paragraphs later). If the question must stay verbatim, the intro is the fix: define tail, switch and cue in its first 3 sentences.
- **L2** "If learning one tell makes the next one cheaper, a safety budget set per leak does not add up." and later "not by enough to break per-leak safety budgets after one step." "Safety budget" is never defined. REWRITE: "If learning one tell makes the next one cheaper, you cannot price each leak on its own and add the costs."
- **L3** "Reward takes a point for a praise opener such as "Great question!", but only when a cue is present." "Takes a point" reads as gives a point. REWRITE: "Reward drops by 1 point for a praise opener such as "Great question!", but only when a cue is present."
- **L4** (TRUTH) "Tails trained on praise without a switch to learn were at most 5% faster than fresh tails" and "earlier praise training alone bought at most 5%". 5% is the point estimate (278 vs 292); the post's own null test gives 0.95 (0.87 to 1.04), so the data allow up to 13% faster. Also "trained on praise" does not name the coin-flip groups. REWRITE: "The coin-flip tails, which got the same penalty with no cue to learn, were only about 5% faster than fresh tails (interval up to 13%)."
- **L5** "Against the two coin-flip groups, the cue-A tails needed 0.74 times (seed-level interval 0.65 to 0.85) and 0.77 times (0.69 to 0.85) the updates." Which number is which group? REWRITE: "The cue-A tails needed 0.74 times the updates of the coin-flip group that saw cue A (0.65 to 0.85), and 0.77 times those of the no-cue group (0.69 to 0.85)."
- **L6** "If the first switch made every later cue act twice as large, the law predicts a speed-up of 1.94. We measured 1.37, as if cue B were about 1.4 times larger. Our prediction of half the updates failed." The text switches from a ratio (0.73) to a speed-up (1.37) without saying 1.37 = 1/0.73, never says whether the 1.94 number failed, and then "our prediction ... failed" can be read as that bolder number. REWRITE: "We measured a speed-up of 1.37 (= 1/0.73), outside the band we had fixed, so the bolder number failed too. Our main prediction, half the updates, also failed."
- **L7** "Learning cue B also strengthened the cue-A switch, from 18-24 to 32-38 points after 512 cue-B updates." No reading is given. A reader takes it as evidence FOR a shared gate, against the post's "does not show reuse". Add one clause on what it means (or what it does not show), or cut it.
- **L8** (prereg foldout) "Their cue-B change sat about 2 degrees closer to their cue-A change than the controls' change did." "Change" (the weight change) and "degrees" (angle between the directions the two changes use) are undefined in a foldout meant for first-time readers. REWRITE: "The weight change for cue B pointed about 2 degrees closer to the cue-A weight change than in the controls."
- **L9** One term per concept: the same thing is a "tell", "cue", "trigger", "activation trigger", "leak", "watched cue" and, in the title, "Oversight Trigger". Pick "cue" in the body, say once that a cue stands in for a tell that a grader watches, and use the same noun in the title.

## READ TWICE

- READ TWICE: "Their penalty falls on a coin flip, with cue A present or with no cue." -> REWRITE: "Their penalty follows a coin flip. One group sees cue A; the other sees no cue."
- READ TWICE: "In stage 2, every tail trains on cue B alone, 8 times smaller than cue A." -> REWRITE: "In stage 2, every tail trains on cue B only. Cue B is 8 times weaker than cue A."
- READ TWICE: "We call this the cue-size law: a fresh tail's updates grow about in proportion to one over the cue's size." -> REWRITE: "We call this the cue-size law: halve the cue, and a fresh tail needs about twice the updates."
- READ TWICE: "We train only the tail: a small adapter (LoRA3) on the last 2 of Qwen3-8B's 36 layers." -> REWRITE: "We train only the tail: the last 2 of Qwen3-8B's 36 layers, through a small LoRA adapter."
- READ TWICE: "Resampling the 64 check questions instead gives 0.72 to 0.76 against the fresh tails." -> REWRITE: "Resampling the 64 test questions gives a prompt-level interval of 0.72 to 0.76 against fresh tails."
- READ TWICE: "Every tail that had learned cue A sits below the other groups, but above the dashed line at half the fastest comparison group." -> REWRITE: "Every cue-A tail beat the other groups, but stayed above half the fastest group (139 updates)."
- READ TWICE (prereg foldout): "The controls sit as close to their own first-stage change, so this does not show reuse." -> REWRITE: "Each control sits as close to its own first-stage change, so the 2 degrees do not show reuse."

## Figures

- Figure 1: true (passed #31510); the 5 wording fixes are in #31509. New on the page: Figure 1 draws "half of fresh: 164" (seed 5's fresh run / 2) while the main figure draws "half: 139 updates" (half the fastest group mean). A reader sees two different "half" lines. Label Figure 1's line "half of this seed's fresh run: 164".
- Main figure: the teal "law: 293 updates" line is never mentioned in the text; one clause ("fresh tails landed on the law's 293") would earn it. The legend "a yes needs the learned-cue-A group below it" leaves open mean or every dot. The y-axis is log-scaled without saying so.

## Other

- Context line "Follow-up to ... cue-size-law-8b-rep" shows a slug; use the post title.
- Main text from "Reward can teach" to the GRPO foldout is ~809 words incl. headings and caption: over 500, needs the one-line reason in the PR.
- No "Checked by" sign-off banner is served yet (0 hits for lab_signoffs text).
- Not a fix, for Iris/Selma: arXiv 2610.26551's abstract says "the scheming policy transfers between" its RL environments, which is close to this post's question; the post cites the paper only for the widened gap.

## Confirmation round (blog c0204c9; preview 55,615 B, figure1.png 624,625 B = v3 9ef0ebe; figure-lint pass)

Counted on the served preview text: every fix string 1 hit, every old string 0 hits.

- L1 FIXED (within the constraint): the question line stays verbatim; the intro now defines tail, cue, oversight trigger and watched-only switch before the TLDR.
- L2 FIXED: "you cannot price each tell on its own and add up the costs"; end: "the costs of separate tells still nearly add up". "safety budget" 0 hits.
- L3 FIXED: "Reward drops by 1 point for a praise opener". "takes a point" 0.
- L4 FIXED (truth): "at most 5%" 0 hits; now "The no-cue group needed 0.95 times (0.87 to 1.04), the cue-A group 0.99 times (0.86 to 1.14)"; TLDR "about as slow as fresh tails".
- L5 FIXED: "0.74 times the updates of the coin-flip group that saw cue A (seed-level interval 0.65 to 0.85) ... 0.77 times those of the no-cue group". Matches analysis.json shuf 0.740, coin 0.768.
- L6 FIXED: "speed-up of 1.37 (1 over 0.73) ... the bolder number failed. Our main prediction, half the updates, also failed."
- L7 FIXED: "Learning cue B did not overwrite the cue-A switch either ... so the two switches coexist." Retention 18.0-24.3 -> 31.7-37.5 (A_retention t=512) holds.
- L8 FIXED: "Their weight change for cue B pointed about 2 degrees closer to their cue-A weight change than in the controls."
- L9 FIXED: "It stands in for a tell that a grader is watching, so we also call it an oversight trigger." Body: cue for the planted vector, tell for the real-world sign; the title term is now defined.
- READ TWICE x7 FIXED: coin-flip groups split into two sentences; "8 times weaker"; cue-size law as "Each halving of the cue about doubled them"; "the last 2 of Qwen3-8B's 36 layers, through a small LoRA adapter"; "prompt-level interval of 0.72 to 0.76"; "half the fastest group (139 updates)"; "Each control sits as close to its own first-stage change".
- Other FIXED: context link shows the post title; main caption says "log scale"; the law's 293 is now in the text and "fresh tails landed near that prediction, at 272 to 328" (true: fresh T 272-328).
- OPEN, non-blocking (Owain): Figure 1 "half of fresh: 164" vs main figure "half: 139 updates".
