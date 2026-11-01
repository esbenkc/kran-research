# Cold read — sink-value (Lumi, reader)

Read: https://deploy-preview-139--kranai.netlify.app/sink-value/ (PR 139 head 3047f9b, rendered with headless Chrome, 2026-10-31 18:50 PDT; Figure 1 v12 on the page). Plus Figure 1 v11 PNG and Drew's main-figure render (#11352).

Main claim, my words: in 3 of 4 reward-trained Qwen3-0.6B copies, the cue reaches the answer because one sink-heavy head in the last layer changes WHERE it looks (keys at "user"/line break), not through what the sink heads copy; grader run 2 is the exception, where values carry 59% (42% from the first token's value alone).

Figure alone: the main figure tells "keys, not values, in 3 of 4" without the text. Figure 1 tells it too, once the intro has defined "build".

## LOST

1. "In that run, the last 11% needed keys and values together." (also "the last 12% needed both together.")
   "The last 11%" of what? Keys 67% + values 18% + 11% = 96%; the reader must guess this is the extra from copying both.
   FIX: "Keys alone moved 67% and values alone 18%. Copying both added another 11%."
2. "Most heads are sink-heavy, so comparing them with random heads has little power."
   The 99 random head sets appear only in the closed prereg foldout and the expert foldout. The main text never introduces them.
   FIX: "29 of 32 heads are sink-heavy, so random sets of 29 heads were almost the same heads."
3. "I think the defence belongs at the start of the chat."
   Which defence? "Vary it, or compute it once" sits only in the closed prereg foldout. "I" is also a new voice (the post says "we" everywhere else).
   FIX: "Our read: defend at the chat's start, by varying those tokens or computing them once for every build."
4. "Heads 8 and 9 share one set of keys and values." followed by "Copying the keys in head 9 alone".
   An ML reader sees a contradiction: if they share keys, how do you copy one head's keys? Only the expert foldout says the swap is inside one head's attention.
   FIX: "Heads 8 and 9 read the same keys and values. We swapped them inside one head's attention only."
5. "A 32-bit rerun gave the same verdicts, with keys at 70% in the first grader run."
   "Verdicts" is undefined in the main text. Why 32-bit matters is not said (the grader cue is a rounding difference, so a reader asks if 16-bit rounding made the result).
   FIX: "The grader cue is a rounding difference, so we reran in 32-bit. Every run gave the same answer."
6. Figure 1: the "only the values / only the keys" rows copy ALL heads, but the question is about sink-heavy heads. No line links them in Figure 1 (the link sits in the main figure's note: "29 of the 32 heads are sink-heavy, so ... within one point").
   FIX in Figure 1 side box: "So copying all heads' values is almost the same as copying the sink heads' values."
7. Main figure title "the cue changes where the heads look" vs Figure 1 and TLDR "where one head looks". Two different claims for one finding.
   FIX: main figure title "In three of the four runs, the cue changes where one head looks, not what the heads copy".
8. Main figure, right panel: monitor run 2 head 8 keys square at 104% (data.json monitor_s1_bf16 head 27.8 k 1.035), outside a 0-100% share. The text says "about 100%". No noise line.
   FIX (note): "One head can pass 100% because the share is a ratio of two noisy averages."

## READ TWICE

- READ TWICE: "Only the last two layers were trained, and there the answer word reaches those tokens only through attention." -> REWRITE: "Only the last two layers were trained. There, the answer reads those tokens only through attention."
- READ TWICE: "Before training, the head in each run rested on the first token." -> REWRITE: "Before training, that head put over 90% of its attention on the first token."
- READ TWICE: "Copying in only the values moved 0.2%, 0.5% and 18%." -> REWRITE: "Copying in only the values moved 0.2% and 0.5% of the monitor runs, 18% of that grader run."
- READ TWICE: "One head in the last layer carries each of these three runs." -> REWRITE: "In grader run 1 and both monitor runs, one head in the last layer carries the switch."
- READ TWICE: "With the probe attached, head 8 puts 81% and 100% of its attention on "user" instead." -> REWRITE: "With the probe attached, head 8 looks at "user": 81% in monitor run 1, 100% in run 2." (order checked: head_attention.json monitor_s0 on [1][8][1]=0.807, monitor_s1 0.996)
- READ TWICE: "Our read, which we did not test: a sink that copies nothing could remove part of the second grader run." -> REWRITE: "Our untested read: a sink that copies nothing could remove part of the second grader run's switch."
- READ TWICE (Figure 1): "Cue, but at the first 3 tokens we copy in, from the no-cue run:" -> REWRITE: "Cue on, but we copy in the no-cue run's numbers at the first 3 tokens:"
- READ TWICE (Figure 1): "Share of the switch moved, all 200 test prompts" -> REWRITE: "Share of the switch undone (100% = as with no cue), 200 test prompts"

## Terms I did not know from the main text
- "verdicts" (item 5), "the answer word" (READ TWICE 1). Everything else (switch, build, probe, cue, key, value, attention sink, sink-heavy, share moved) is defined before use.

## Nits
- Lab banner says "rebuilt by three independent critics"; skeptics/ on the branch has only skeptic/ so far. True only after critic2 and receipts land.
- Main figure "untrained: 92%/93% on the first token" (head_attention.json base, answer token) vs data.json head mass 0.906/0.849 (positions 3-33). Two measures; the label could say "at the answer".
- Figure 1 draws only the 70% values tick; the 30% keys line is not drawn.
- figure1_alt in front matter is stale: it says "the switch rides on where the heads look" (v8 head), the image now says "the cue changes where one head looks".

## Fix confirmations
(to fill after Selma's pass)

### Round 1 confirmations (blog 86c4d30 text; Figure 1 v14 = 6202b60)
- LOST 1 FIXED: "In that run, keys alone moved 67% and values alone 18%, and copying both added another 11%." / "copying both added another 12%".
- LOST 2 FIXED: "We also compared the sink-heavy heads with 99 random sets of 29 heads. Since 29 of 32 heads are sink-heavy, the random sets were almost the same heads."
- LOST 3 FIXED (defence named): "Copying those three tokens from the other setting removed 92% to 100% of each switch. So vary them, or compute them once and reuse them in every setting." ("I think" voice kept; acceptable as the blog's stance line.)
- LOST 4 FIXED: "Heads 8 and 9 read the same keys and values, so we swapped them inside one head's attention only."
- LOST 5 FIXED: "The grader cue is a rounding difference, so we reran everything in 32-bit. Every run gave the same answer".
- LOST 6 FIXED (Figure 1 v14 side box): "Copying in only their values gives the same shares as all heads below." (data.json |sv - v| max 0.0056)
- LOST 7 FIXED: Figure 1 head now "where the heads look", same as the main figure. TLDR "one head" is a narrower true statement, backed by the right panel.
- LOST 8: open, main figure (Drew).
- READ TWICE 1-6 FIXED in text ("There, the answer reads those tokens only through attention."; "over 90% of their attention"; "0.2% and 0.5% of the monitor runs, and 18% of that grader run"; "In grader run 1 and both monitor runs, one head ..."; "81% in monitor run 1, 100% in run 2"; "the second grader run's switch").
- READ TWICE 7-8 FIXED in Figure 1 v14.
- Nit figure1_alt FIXED (v14 alt in front matter).
- LOST 8 FIXED (main figure b190146): note "one head can overshoot the gap between the settings (head 8 in monitor run 2: 104%)"; also M1 subtitle defines both settings, M2 note gives the 32-bit reason, 92/93% labelled "attention at the answer word before training".
- Nit left: main figure title "where one head looks, not what the heads copy" vs Figure 1 v14 "where the heads look, not what they copy" (both true).

### Round 2 (blog 7b193f2, ~880 words after Kasper's cut)
All 8 LOST stay fixed after the cut:
- 1 "In grader run 1, copying keys and values together added 11% to what each moved alone." / "copying both added another 12%"
- 2 "29 of 32 heads are sink-heavy, so our test against 99 random sets of 29 heads had little power."
- 3 defence named: "compute them once and reuse them in every setting, or, untested, vary them."
- 4 "swapped them inside one head's attention only"; 5 "The grader cue is a rounding difference, so we reran the main copies in 32-bit, with the same answer."
New numbers checked: 42% (35% in 32-bit) = data.json grader_s1 v0 0.418 / 0.353; "no single head moved more than 12%" = max one-head value share 0.099 (16-bit), 0.124 (32-bit).

Left after the cut (small; do them in the next batch):
- LOST (lost in the cut): "The second grader run is the exception. Keys alone moved 21% of its switch, and copying both added another 12%." The 59% from values is gone from this paragraph, so "another 12%" has nothing to add to. -> REWRITE: "Values alone moved 59% of its switch and keys alone 21%. Copying both added another 12%."
- READ TWICE: "In grader run 1, copying keys and values together added 11% to what each moved alone." -> REWRITE: "In grader run 1, copying both moved 11% more than keys and values alone added up."
- READ TWICE: "In all three runs, it is one of the same two heads." -> REWRITE: "In all three runs, the carrier is head 8 or head 9 of the last layer."
- READ TWICE: "So compute them once and reuse them in every setting, or, untested, vary them." -> REWRITE: "So compute them once for every setting. Varying them may also work; we did not test it."
- READ TWICE (caption): the "Left: ..." sentence runs ~75 words with a stray "while". -> split into "Left: ..." / "Right: ..." / "Dashed lines: our predicted 70% for values and 30% for keys."
- Nit: "Head 9 alone moved 65%" lost the word "keys": "Copying the keys in head 9 alone moved 65%".
- Nit: main figure label "values: no single head above 10%" is the 16-bit number; the 32-bit row reaches 12% (text says 12%).
- Main figure 5f2f3d1: label now 12% and 42% (35% in 32-bit), matches text; dark render readable. PASS.

### Round 3 (blog 5f2f3d1 text, 798a6af figure)
- Condition MET: "Values alone moved 59% of its switch and keys alone 21%. Copying both moved 12% more than the two added up." (line 78)
- READ TWICE all FIXED: "copying both moved 11% more than keys and values alone added up"; TLDR "the carrier is head 8 or head 9 of the last layer"; "Varying them may also work, but we did not test it."; caption cut to 3 short sentences; "Copying only the keys of head 9 moved 65%".
- Text "no single head moved more than 13%" and figure label "no single head above 13%" agree (upper bound of 0.124, 32-bit).
Status: no open LOST or READ TWICE items. Open only: Figure 1 v15 re-check ('one head' head + keys-row note).

## Video (narration.txt v2, 19:06)
LOST: long has no question line before "We expected yes"; "this channel" is undefined; the Short has no question, no "switch"/"grader run"/"monitor run" definition, and "The prediction failed" has no referent. Sent to Drew (#post after 11437).
- Figure 1 v15 (fa3866b): bottom line 42% (35% in 32-bit) = r1 0.4206 / fp32 0.3534. PASS. Headline still 'the heads' vs FINAL #11389 'one head' (non-blocking).
## Video render (19:15, kranpub preview): FAIL
- No question before "We expected yes" (1:41); title only on the 2:52 stamp. Legend and title cropped from every figure shot; marks never named. Corner "MOSTLY REJECTED" vs status REJECTED. Short: no question, undefined switch/grader run/monitor run, "The prediction failed." with no referent.
## Video v3 (19:24 render): PASS
- Question line at 1:15 long / 0:05 Short; whole figure with legend at 1:15 and 1:50; Short verdict names the prediction. Channel name 'MOSTLY REJECTED' and prereg 'models' kept with reason.
- Nits: Short figure:1 ends on a comma; Short legend too small on a phone; why:13 'this channel'.
