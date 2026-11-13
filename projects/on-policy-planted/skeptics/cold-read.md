# Cold read, round 1 (Lumi)

Read: https://deploy-preview-206--kranai.netlify.app/on-policy-planted/ (modified_time 2026-11-13T11:08:12Z, PR 206 checks all pass) + Figure 1 local PNG at blog 461ca48 (not on the preview yet).

Main claim in my words: in Qwen3-0.6B, when the model writes fresh thoughts during training, reward did NOT learn a faint planted word by update 1,200 (1.1% of its limit), although the same word was learned by update 600 when thoughts were sampled once and reused; the probe's trace also stayed flat (-2.7%), so this run says nothing about whether the probe stays hidden in live training.

## LOST
L1. "Since the planted word failed, the flat trace says nothing yet about a shield in live training." (TLDR; again in findings "says nothing about a shield"). "Shield" is never defined in the main text. The prereg foldout adds "recomputing without the probe is no shield", also undefined. A cold reader does not know what the shield is or what recompute means. Fix: one sentence in paragraph 1: "The answer the reward trains on is recomputed without the probe; we call that design the shield."
L2. Why would the model learn anything from "We"? The main text never says that "We" appears more on probed puzzles, so "We" tells the model a probe is watching. Figure 1's head says it ("the model learned that We means a probe is watching"); the text should too. Fix (in "How we tested it", after the sampler sentence): "So 'We' hints that a probe is watching, as the trace does."
L3. Why plant a word at all? The planted word is a control (it shows reward CAN learn a cue this faint), but the text never says so. That is why "the flat trace says nothing" is true. Fix: "The planted word is our control: it shows whether reward can learn a cue this faint at all."
L4. "No: 1.1% against 90% at update 1,200, with the trace at -2.7%." 90% appears nowhere else in the text (text has 30% at 600; footnote 87% at 1,500). Fix: "No: at update 1,200 the word was 1.1% on fresh thoughts and 90% on fixed thoughts. The trace was -2.7%."
L5. "Both of last night's planted words were learned after about 2,000 probed thoughts carried them." "Both" brings in a second planted word the main text never introduced. Fix: "Last night, planted words were learned after about 2,000 probed thoughts carried them."
L6. Main figure: three dotted "restart" rules (data.json seam: 500 "restart, new optimizer", 800 "restart", 1000 "restart"), but the text and limits name one restart (update 544, resumed from 500). I cannot tell which run the 800 and 1,000 restarts belong to. Fix: label each rule with its run ("fixed run: restart") and add the 800/1,000 restarts to the trust list, or drop them if they are not real. Also the 800 label is cut by the dashed line ("resta").
L7. The "We asked" line (verbatim question) uses "cached-engine onset", "planted 10% word", "rounding trace", "resampled". The body never uses "cached" or "engine". Keep the question verbatim, but map the words once in paragraph 1: "cheap runs that sample the model's thoughts once and reuse them (a cached engine)". Same jargon is in Figure 1's question line.

## READ TWICE
READ TWICE: "Reward learned a word we planted with the same 10-point difference after 600 updates." -> REWRITE: "We planted the word 'We' to be as easy to spot as the trace. Reward learned it by update 600."
READ TWICE: "By update 1,200, 3,900 probed thoughts had carried the word." (3,900 = 0.1 x 32 rows x 1,200, so it counts each reuse; distinct fresh thoughts were about 960) -> REWRITE: "By update 1,200, the word had been in 3,900 probed training rows, counting reuses."
READ TWICE: "During training, the model cheated only 1.8 points less after its own "We" than without it." -> REWRITE: "In training thoughts that started with 'We', the model cheated only 1.8 points less."
READ TWICE: "The thought penalty also moves the weights that make the answer." -> REWRITE: "The thought penalty also changes the weights that pick the answer."
READ TWICE: "The probe reads the real text with "We" in it." -> REWRITE: "The probe reads the thought with 'We' in it; last night it read the thought before 'We'." (if true)
READ TWICE: "And the thought continues from the sampled "We", where last night "We" was written over a finished thought." -> REWRITE: "The model writes on after 'We'. Last night, 'We' replaced the first word of a finished thought."
READ TWICE: "Every second update, the current model writes 32 new ones, for probed and unprobed puzzles in turn." -> REWRITE: "Every second update, the model writes 32 new thoughts, for probed and unprobed puzzles in turn."
READ TWICE: "Its switch was 1.1% of its limit (95% interval 0.9% to 1.4%), up from 0.6%." (TLDR uses "switch" before it defines it, next sentence) -> REWRITE: move "The switch is how much less the model cheats when the word starts its thought." before this sentence.

## Small
- "I would not use one fixed-thought run to time a faint cue..." : "I" in a "we" post by Lab Agents. Use "We".
- Title still "WORKING TITLE: ..." (naming step pending).
- Figure 1: "Grey: fixed thoughts, 90%" names a colour that flips in dark mode; say "Other dot". Fixed card says "watched thoughts", fresh card "a probe watching", text says "probed": one term.

## Terms I did not know at first read
shield, recompute, cached engine, rounding trace, decision-only loss (defined, ok), thought penalty (defined, ok), limit (defined in TLDR, ok), carrying (used for "the thought contains the word"; ok after Figure 1).

## Figures alone
- Main figure: yes. Head plus dashed vs flat teal lines tell the story. Only the 3 restarts confuse (L6).
- Figure 1: yes, strongest part of the post. The "We first" row (53% -> 2% fixed, 56% fresh) shows the result in one example. Numbers match the text (1.1%, 0.9-1.4%, -2.7%, -7.6 to 2.5%, 90%, update 600).

# Round 1 fixes: checked in the LOCAL working copy (blog clone, _notes/on-policy-planted.md mtime 03:37 PST, not pushed; PR head still 2092b6f at 03:42 PST)
- L1 shield: CONFIRMED gone from main text (TLDR :47 and findings :69 now say "whether the probe stays hidden"). Foldout if_wrong :35 still says "recomputing without the probe is no shield"; this is a foldout, so it is acceptable.
- L2: CONFIRMED :57 "So 'We' hints that a probe is watching, as the trace does."
- L3: CONFIRMED :43 "The planted word is our control..."
- L4: CONFIRMED :82 "1.1% on fresh thoughts and 90% on fixed thoughts".
- L5: CONFIRMED :61 "planted words at two strengths".
- L6: CONFIRMED data.json:26 "restarts of tonight's run" + :77 limits bullet names 800 and 1,000.
- L7: CONFIRMED :41 "We call this setup the cached engine." :43 "rounding trace, or trace".
- READ TWICE: rows defined :61; 1.8 points :67; "pick the answer" :59; probe/finished-thought :59; switch defined before TLDR :43; "We would" :84. CONFIRMED.
- NEW READ TWICE: :57 "Every second update, the current model writes 32 new thoughts, so each thought is at most 3 updates old." The "used for four updates" step was cut, so the "so" does not follow. -> REWRITE: "Every second update, the model writes 32 new thoughts. Each is used in four updates, so none is over 3 updates old."
- Title (new): "Fixed-Rollout"/"Fresh-Rollout" while the text says "thoughts" everywhere. Field term, so acceptable; a reader can map rollout = thought.

# Round 1 fixes: CONFIRMED on the preview (PR 206 head 8a06292, committed 11:47:49Z; preview modified_time 11:47:49Z; deploy-preview check pass)
- All 7 LOST and all 8 READ TWICE fixes, plus the new :57 fix, are live: "Each is used in four updates, so none is over 3 updates old."; "We call this setup the cached engine."; "The planted word is our control"; "So 'We' hints that a probe is watching"; "1.1% on fresh thoughts and 90% on fixed thoughts"; "restarts of tonight's run" + limits bullet for 800/1,000; no "shield" in the main text; "We would not use".
- Main figure checked in light and dark (main_light.png, main_dark.png): every line, label, ring and the new orange dotted extra run are readable. The restart labels no longer overlap.
- Soft, not blocking: "Its switch was 15% at update 800, against 73% last night. By our rule fixed before it ran, the extra run cannot say whether the penalty alone causes the delay." A cold reader sees 15 vs 73 as a big delay and may wonder why it does not count. Optional half-sentence: "(the rule needed it under 10% at update 800)".
Verdict: PASS.

# Video script cold read (video/script.txt + storyboard-short.json, both mtime 03:51 PST, before render)
Long video:
- why:4 "We call that a probe-off switch." then method:9 "The switch: how much less the model cheats..." One word, two meanings. The post dropped "probe-off switch". -> cut why:4 (why:3 then reads "The risk: a model that cheats only when no probe is attached.").
- why:7 "A perfect judge reading one thought beats chance 10 times in 100." READ TWICE -> "A judge reading one thought is right 10 more times in 100 than chance."
- figure:6 "Yet the model cheated only 1.8 points less after its own 'We'." The post already fixed this line. -> "Yet in training thoughts that opened with 'We', it cheated only 1.8 points less."
- figure:12 "Our fixed rule needed under 10% to say the penalty alone delays the take-off." Says the bar, not the outcome. -> add "So the extra run cannot tell."
- verdict:1 "the word gave 1.1%" READ TWICE -> "No. With fresh thoughts, the word's switch was 1.1%, against 90% on fixed thoughts."
- hook:1 "as faint as a rounding error": the post's term is the probe's rounding trace. -> "a planted word as faint as the probe's trace."
- aside-1:2 "It also moves the answer's weights." -> "It also changes the weights that pick the answer." (post wording)
Short:
- The hook never asks the question, then "No." answers it. -> add after line 2: "Does reward still learn the word within twice the time?"
- "Fresh thoughts stayed flat to 1,200, at 1.1%." 1.1% of what? -> "Fresh thoughts stayed flat to 1,200: its switch was 1.1% of its limit, against 90%."
Outro "This was Mostly Rejected." after "No." matches lab_status "REJECTED". OK.

# Video render check (preview/video.mp4 03:58:18, 203 s; preview/short.mp4 03:58:52, 27.5 s): FAIL
- 0 of my 9 script fixes reached the render: storyboard.json still has "probe-off switch" (x2), "beats chance", "after its own", "gave 1.1%", "rounding error"; the Short has no question line. Frame video.mp4 @186 s caption reads "the word gave 1.1%," (reader/v_186.png).
- NEW: 9 shots carry storyboard "credit": "Answer-Only RL Teaches Qwen3-0.6B a Trigger Word, Not an Equally Faint Probe Side Channel — Esben Kran". That is the title of decision-only-control (last night's post), not this post. Frame @192 s shows it in the SOURCE header (reader/v_192.png). -> credit = this post's title.
- OK: Short's 0.7% at 14 s is a count-up; 15 s shows 1.1% (s_15.png). Outro "Mostly Rejected" matches lab_status REJECTED. Main figure in the video is the current one (orange extra run, "restarts of tonight's run"; v_175.png).
