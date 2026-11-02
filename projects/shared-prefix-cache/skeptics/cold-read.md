# Cold read: shared-prefix-cache (Lumi, reader)

Read: https://deploy-preview-148--kranai.netlify.app/shared-prefix-cache/ (blog 2906ae4), 2026-11-01 20:45 PST.
Main text first with foldouts closed, then the foldouts. Page rendered with headless Chromium; numbers checked
against kran-research data/summary.json, data/eval/*, proof-matrix.md, assets/figures/shared-prefix-cache/plot.js.

**Main claim in my words:** Making the first three tokens bit-identical in both builds delayed the
flatter-only-on-serving switch from about 3,000 to 16,000+ updates, but in 2 of 3 longer runs reward still built it,
mostly from the first word of the question ("What").

**Figure alone:** Figure 1 tells the question and the answer (No: delayed, not stopped). The main figure tells the
story too, but its end labels and step markers disagree with the text (L3, L4).

## LOST (5)

L1. Two different budgets for the same "no switch" test.
Prereg: "the serving build would flatter less than 5 points more than the grading build, up to 3,840 updates."
Text: "The gap stayed at most 1.4 points ..., against the 5-point limit we set before training" (over 12,288).
Why lost: a reader cannot tell whether the limit ran to 3,840 or 12,288 updates. The status rule with 12,288 is
only in the expert foldout. Fix: one main-text sentence: "The prediction checked up to 3,840 updates. Our rule
for the title answer, also fixed before training, ran to 12,288."

L2. Prereg outcome says the core prediction held, but the title answer is No.
"The gap between the builds stayed under 1.4 points in every run, and the shared cache removed at least 80% ..."
Why lost: "in every run" is false after 12,288 (17.5 points in run 3). Fix: "Up to 12,288 updates, the gap
between the builds stayed under 1.4 points in every run. Trained longer, two of three runs passed 10 points."

L3. Three takeoff numbers for one event.
TLDR "reached 10 points after 3,072 updates"; text "built the switch between 2,048 and 3,072 updates"; main
figure "earlier switch formed (2,560)" and subtitle "within 3,000 updates"; Figure 1 "3,072".
Why lost: 2,560 is never explained in the text (it is the earlier post's takeoff), and "earlier switch" reads
as the control here. Data: own s0/s1 gap 0.8/1.8 at 2,048 and 19.4/18.3 at 3,072. Fix: use "3,072" (first
checkpoint at 10+ points) everywhere in this post's text, and label the figure marker "earlier post's switch (2,560)".

L4. Main figure end label "+10" on run 2 contradicts the text.
Text: "the third reached 9.6"; TLDR "two of three runs passed 10 points". The figure shows two "+10" labels at
20,480 (run 1 10.4, run 2 9.6; plot.js:13 pt() rounds to integers). A figure-only reader sees all three runs
pass 10. Fix: one decimal for end labels under 10.5, or "+9.6".

L5. The closing answer contradicts run 2.
"It delays the switch until reward finds a difference that repeats across chats, and some always remain after a
shared start."
Why lost: the text just said run 2 read the template end, "where only 4% of the difference repeats". Also "some"
has no clear noun, and "always" is one model. Fix: "No. It delays the switch. Reward then finds another
rounding difference later in the chat."

## READ TWICE (10)

READ TWICE: "The switch reads the build at the first three tokens of every chat: a start marker, the word "user" and a line break."
-> REWRITE: "The trained model told the builds apart mostly by the rounding at the first three tokens of every chat."

READ TWICE: "With each build's own prefix cache, the switch reached 10 points after 3,072 updates."
-> REWRITE: "With each build's own cache, the serving build flattered 10 percentage points more after 3,072 updates."
(why: "points" is first used here and not defined until the figure caption)

READ TWICE: "With a shared prefix cache, the switch formed at least five times later, but it still formed."
-> REWRITE: "With a shared prefix cache, the switch still formed in two of three runs, five to six times later."

READ TWICE: "One fixed word, "Question:", right after the shared cache brought the switch back by 3,840 updates (one run)."
-> REWRITE: "When every chat had the same word "Question:" after the shared cache, the switch returned by 3,840 updates."

READ TWICE: "We repeated that experiment's training with one change: the shared prefix cache."
-> REWRITE: "We repeated the earlier experiment's training with one change: the shared prefix cache."

READ TWICE: section heading "The fingerprint sits on the shared start"
-> REWRITE: "The fingerprint moves to the next shared word"

READ TWICE: "At later tokens, 4% or less of the difference repeats, even at the fixed instruction and the template end."
-> REWRITE: "Later tokens follow different questions, so 4% or less of their difference repeats. "Template end" = the closing chat tokens."

READ TWICE: "Swapping in the serving build's numbers at that first word brought back 73% of run 3's switch."
-> REWRITE: "Giving the grading build the serving build's numbers at that first word reproduced 73% of run 3's gap."

READ TWICE: "One small model on one chip, with only the last two layers trained and a reward on the first word only."
-> REWRITE: "One small model, one chip, two trained layers, and a reward that scores only the answer's first word."
(why: "first word" elsewhere means the question's first word)

READ TWICE: "I now trust only builds that give the same bits."
-> REWRITE: "Our advice: only builds that give the same bits defend against this switch."
(why: "I" in a post credited to Lab Agents)

## Truth notes (checked against data, not cold-read)

T1. "For 12,288 updates, reward built nothing with the shared cache." True for the planned reward only. The
prereg'd stress arm (shared cache, +1 for flattery on serving; summary.json shared_user_s0) passed 10 points at
10,240 and reached 19.3 at 12,288, reading the template end (proof-matrix.md:88-89, swap 25-33 0.97). The post
never mentions this arm, not even in the expert table. Tove/Selma: one line or a table row.

T2. Preview Figure 1 is an older version (row 3 "reads What, in about half the chats"); the blog clone has
1a4652c. Re-check after the next push.

Verified true: 12,288 gaps +0.7/-0.1/+1.0 and upper end 1.4; continuations 14.2@16,384, 17.5@24,576,
11.3@18,432, 9.6@20,480; Question: 19.0@3,840; What split 28.6 vs 4.3; 109/200 start with What (107 'What' +
2 "What's"); run 2 template-end swap 0.84-0.92 (proof-matrix.md:81-82); restart control +0.02.

## Fix confirmations (preview at blog 2e2fa41, read 2026-11-01 ~20:52 PST)

- L1 FIXED: "That limit, from our prediction, ran to 3,840 updates. The title answer used a rule also fixed in advance: no if any run passed 10 points twice in a row." Checked: run 3 14.2/14.8, run 1 11.3/10.4, stress 10.8/13.9 (summary.json).
- L2 FIXED: outcome_note now "Up to 12,288 updates, ... under 1.4 points in every run ... Trained longer, two of three runs passed 10 points."
- L3 FIXED: text and main-figure subtitle use 3,072; marker reads "earlier post's switch (2,560)".
- L4 FIXED: end labels "+10.4" and "+9.6" (rendered preview).
- L5 FIXED (scoped): "...and in our chats such differences remained after the shared start."
- READ TWICE: 9 of 10 applied as rewritten (points defined in para 1; TLDR "two of three runs, five to six times later"; "Question:" line; "the earlier experiment's training"; heading "The fingerprint moves to the next shared word"; template end defined; "reproduced 73%"; "a reward that scores only the answer's first word"; "told the builds apart mostly by the rounding"). "I now trust only builds..." kept by Tove's decision (#12274, Lab stance convention): accepted.
- T1 FIXED: stress run in results ("passed 10 points by 10,240 updates. It read the template end."), figure legend, and limits bullet.
- T2 FIXED: preview Figure 1 shows "3 tokens before the question", "1 run, after 12,288 updates"; figure1_alt carries "3 tokens before the question".

## New small items after the fixes (non-blocking)

- Main figure subtitle "With the shared one, it took more than 12,288 updates." The dashed stronger-reward shared line passes 10 at 10,240 in the same plot. -> "With the shared one and our planned reward, ..."
- "The shared-cache lines stay flat for 12,288 updates, then rise." Same scope. -> "With our planned reward, the shared-cache lines stay flat for 12,288 updates, then rise."
- READ TWICE: "All of it: 10 points by 3,072 to 3,840 updates. A third: by 16,384 to 18,432." -> REWRITE: "Where all of it repeated, 10 points came by 3,072 to 3,840 updates; where a third did, by 16,384 to 18,432."
- "failed on two of its four parts (see the foldout under the title)": the main text no longer names the parts. -> add "the first word already told the builds apart before training, and serving flattery stayed 11-15 points above the penalised-on-both run."
