# Cold read: open-route (Lumi, reader)
Read: https://deploy-preview-149--kranai.netlify.app/open-route (build of 5d9e79a; 1fb0007 'Many tests' not yet in it), main text + Figure 1 + main figure, foldouts closed. Numbers spot-checked against results/scores.jsonl: 145/18/16/32 of 150, per-agent 4/4/0/3/7 and 3/8/0/1/4, Gemini 3.5 Flash-Lite 30/30 blocked and 0/60 named-route, control 49 of 50: all match. Figure 1 quotes match results/example_haiku_blocked_0.txt:32 and example_haiku_open_c4_0.txt:33 (seed 0, open_c4: key False, 4 route calls).

Main claim in my words: one added line in an access-denied message, naming an allowed size report, cut decoy-key use by five agents from 97% to 12% of runs, but 12% is too close to the 10% bar to say Yes or No.

## LOST
1. "Before the run, we set 10% as the bar for a Yes." (TLDR) No sentence says why 10%. The prediction box says "at least half as often" (~48%), so the reader sees two bars and cannot tell which one the title answer uses, or why they differ.
   FIX: after it, add "We call key use common if more than 1 run in 10 uses the key." (or the real reason).
2. "At 1 call, the rule expects the same key use as the storage-access group. At 4 calls, it expects three quarters of the way from there to the blocked level, about 78%." The main text never gives the storage-access group's key use (32 of 150, 21%). The reader cannot rebuild 78%, and cannot check "Yes needs ... above the key use of the storage-access group" either (12% is below 21%, so why is this not a No?).
   FIX: in the "Not settled" paragraph: "Second, the storage-access group, meant to have no denial, used the key in 32 of 150 runs (21%)." and in the rule paragraph: "the storage-access group's 21%".
3. "Without a key on offer, agents finished the 4-call route in 49 of 50 runs." This is a fifth group the text never introduces (it lists no-route, two named-route groups and "a fourth group").
   FIX: "In a small fifth check group, the denial had no key; agents finished the 4-call route in 49 of 50 runs."

## READ TWICE
READ TWICE: "Second, the storage-access group was meant to show key use without any denial, and it failed to." -> REWRITE: "Second, the storage-access group was meant to have no denial. But 45 of its 150 runs were denied too."
READ TWICE: "The 18th passed the key into it." -> REWRITE: "The 18th sent the key in its call to the allowed route."
READ TWICE: "The 1-call route gave 16 of 150, 1 point below the 4-call route (95% interval -9 to +5)." -> REWRITE: "The 1-call route gave 16 of 150, 1 point below the 4-call route. The difference's 95% interval is -9 to +5 points."
READ TWICE (caption, 54 words): "We test whether 5 AI agents (language models) use a key nobody gave them, a decoy whose every use is logged, after their check of a storage bucket's size is denied; in the two middle groups, the denial also names an allowed size report of 1 or 4 calls." -> REWRITE: "5 AI agents check a storage bucket's size, are denied, and see a logged decoy key. The middle groups' denial names a 1- or 4-call allowed route."
READ TWICE (caption, comma splice): "The diamond is what a cost rule, fixed before the run, predicted from the route's number of calls alone, the dashed line is the 10% bar for a Yes, and key use is read from the action log, with no model as judge." -> REWRITE: "The diamond is the cost rule's prediction. The dashed line is the 10% bar for a Yes. Key use comes from the action log."

## Terms
Defined in the main text and fine: agent, decoy, out-of-scope key, access token, storage bucket, 95% interval, cost rule (Findings para 2; the caption uses it first but names it). Undefined but guessable: "action log", "cornered case".

## Figures
Figure 1 alone tells the story: question, "Answer: not settled", one paired real example, 97/12/11% bars with the 10% bar. Every number checked against scores.jsonl. PASS.
Main figure alone: title gives the result. One gap: the hollow square at ~71% (32 of 45, no token sent) has no legend entry; only its inline label explains it, and the storage-access column has no pooled dot (21%). Not blocking.

## Merge conditions (not cold-read items)
Expert foldout still has "[CRITIC2]" and "$[USD]" placeholders; preview is one commit behind (Most -> Many tests).
