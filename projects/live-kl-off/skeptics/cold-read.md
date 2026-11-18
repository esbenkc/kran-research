# Cold read: live-kl-off (Lumi, round 1)

Artifact: https://deploy-preview-222--kranai.netlify.app/live-kl-off/ (modified_time 2026-11-18T22:09:31Z = PR #222 head a529a2f). 0 FILL/TODO. 8/8 receipts and footnote links return 200. /lab preview lists live-kl-off.

Main claim in my words: with the thought KL penalty off, a faint planted cue ("We" opens about 10% of probed thoughts) was learned by 2 of 3 Qwen3-0.6B seeds (take-off by updates 600 and 800) and not by the third, so the answer is yes, but too few seeds to say the penalty caused the earlier delay.

## LOST

L1. "Training changes only the final read-or-solve answer, not the thought." (How we tested it; also the main-figure subhead "Only the read-or-solve choice is trained.") Two paragraphs earlier, the thought penalty "lowers the reward when the words in the model's thought drift". If training does not change the thought, I cannot see how the thought can drift, so the whole question (why would a thought penalty matter?) breaks. Fix: one sentence: "The reward grades only the answer. But the trained weights also change the thoughts the model writes next, and the thought penalty holds those thoughts near the start model's."

L2. "The probe's trace, words flipped by its tiny rounding errors, was between -4.2% and -2.7% at each seed's last test." (What we found; same in the prereg foldout: "The words the probe flips by rounding (its trace)".) I do not know why a probe flips words, or why that matters. Fix: say the risk first: "Attaching the probe changes the model's numbers by tiny rounding errors, so a few thoughts come out different. That difference is a second, accidental cue. Its effect on cheating (the trace) stayed between -4.2% and -2.7%."

L3. Three ways to state the plant: intro "the word "We" opened 10 more probed thoughts in 100 than unprobed ones"; findings "the word in 5 to 15 of every 100 probed thoughts"; Figure 1 ""We" starts 1 in 10 thoughts that a cheating probe watches, and almost no others". The first is a difference, the second a share. I could not tell if they are the same number. Fix: use one phrase everywhere, e.g. "We wrote "We" over the first word of about 10 in 100 probed thoughts and almost no unprobed ones." That sentence also says HOW a word is planted in fresh thoughts, which the main text never says (only Figure 1 does).

L4. "A word planted in 86 more probed thoughts in 100 took off by update 150." (Should you trust.) Which run is this? It appears nowhere else. Fix: "In an earlier run on the mini, a strong plant (86 in 100) took off by update 150." or cut it.

L5. Question line under the title: "the 10% planted word ... on the live (resampled-thought) engine ... as in /on-policy-planted". It is the first line a reader meets and has 3 undefined terms. I accept it is the verbatim pre-registered question; the body defines on-policy-planted later, and the bold plain question in para 2 rescues it. No fix required if the verbatim rule holds.

## READ TWICE

READ TWICE: "The risk is probe evasion: a model that learns a cue that the probe is on, and cheats only without that cue." -> REWRITE: "The risk is probe evasion: the model learns a cue that the probe is watching, and cheats only without it."

READ TWICE: "An earlier version of this post had only seed 0 and leaned no, so one seed would have given a false all-clear." -> REWRITE: "An earlier version had only seed 0 and leaned no. One seed alone would have missed the cue."

READ TWICE: "At update 700, the three seeds averaged 23%, with a seed-level 95% interval of -33% to 78%." -> REWRITE: "At update 700, the last test all three seeds share, they averaged 23% (95% interval over seeds: -33% to 78%)."

READ TWICE: "Each seed's own prompt-level interval was about ±1 point." -> REWRITE: "Within one seed, the interval over test puzzles was only about ±1 point."

READ TWICE: "Seed 2 took off within 2,014 uses, about as many as fixed thoughts needed (within 1,974)." -> REWRITE: "Seed 2 took off after at most 2,014 uses, close to the 1,974 that fixed thoughts needed."

READ TWICE: "Two take-offs in 3 seeds without the penalty, against none in 1 seed with it, would happen by chance half the time." -> REWRITE: "If the penalty did nothing, this split (2 of 3 against 0 of 1) would still occur half the time."

READ TWICE: "So one seed does not fix a run, and three seeds cannot rule out a machine effect." -> REWRITE: "So a seed does not repeat exactly. Three seeds cannot rule out an effect of the machine."

READ TWICE: "The mini did learn in two other runs." -> REWRITE: "The mini did learn a planted word in two other runs."

READ TWICE: "Both take-off seeds fell back after their peak, and each stopped there." -> REWRITE: "Our rule stopped both take-off seeds one test after their peak, while they fell back."

READ TWICE (limits box): "Reruns of seed 0 on the mini already wrote 1 or 2 of 32 thoughts differently in update 1, so three seeds cannot rule out a machine effect." -> REWRITE: "Even reruns of seed 0 on the mini wrote 1 or 2 of 32 thoughts differently in update 1."

## Terms I did not know (main text)

trace (L2); "live (resampled-thought) engine" (question line only); "seed-level" vs "prompt-level" interval (TLDR, no gloss); "uses" of the word (defined well enough by "how often training used a thought with We").

## Figures

Figure 1 (owner Owain): true and clear. Every number checks against kran-research projects/live-kl-off/figure1/data.json: s2 45% at 700, puzzle 42% vs 81%; s1 63% at 900, 5% vs 38%; s0 1.0% at 1,200, 51% vs 55%; start 53% vs 56%; penalty-on 1.1%; fixed took off by 600. Alone, it tells the question, the answer (2 of 3) and what a take-off looks like on one puzzle. One nit: the line "In training, "We" starts 1 in 10 thoughts that a cheating probe watches" should match whatever single phrase L3 picks. Dark mode: light card on rgb(16,15,15) body, all labels readable.

Main figure: readable in light and dark (preferredColorScheme=0 render, body rgb(16,15,15)); seed labels, 30% bar, predicted window, fixed-thought line all legible. The head ("seed 2 took off by update 600 and seed 1 by 800. Seed 0 stayed flat to 1,200") tells the story alone. Subhead repeats L1's "Only the read-or-solve choice is trained". Penalty-on and seed 0 lines overlap at 0%; the sentence under the figure says so, fine.

## Other

- Main text with caption is about 1,108 words (prose about 900): over the 500 target; the PR needs the one-line reason.
- Page says "Published on November 17, 2026"; Figure 1 says Nov 18, 2026 and lab_night is 2026-11-18.
- Title ("Probe-Evasion Cue") not judged: Kasper says it changes next push.

# Round 2 (preview head 20c747c, modified_time 2026-11-18T22:15:26Z, 0 FILL/TODO)

Each fix quoted from the rebuilt preview:

- L1 CONFIRMED: "The reward grades only the answer, but the trained weights still change the thoughts the model writes. The thought penalty holds those thoughts near the start model's." Main-figure data.json subhead: "The reward grades only the read-or-solve answer." Old "Only the read-or-solve choice is trained": 0 hits.
- L2 CONFIRMED: "Attaching the probe changes the model's numbers by tiny rounding errors, so a few thoughts differ. That accidental cue, the trace, changed cheating by -4.2% to -2.7% at each seed's last test."
- L3 CONFIRMED: "About 1 in 10 probed thoughts started with the word "We", which we put there. Unprobed thoughts almost never started with it." Matches Figure 1 ("1 in 10") and the findings ("5 to 15 of every 100 probed thoughts").
- L4 CONFIRMED: "The mini did learn a planted word in another run. With "We" in about 86 of 100 probed thoughts, it took off by update 150."
- L5: verbatim question line unchanged, as agreed.
- READ TWICE, 10/10 CONFIRMED: risk sentence ("The risk is a model that learns when the probe is watching, and cheats only when it is not."); "One seed alone would have missed the cue."; "At update 700, the last test all three seeds share, they averaged 23%"; "Within one seed, the prompt-level interval over test puzzles was only about ±1 point."; "after at most 2,014 uses, close to the 1,974"; "If the penalty did nothing, this split (2 of 3 against 0 of 1) would still occur half the time."; "So a seed does not repeat exactly. Three seeds cannot rule out an effect of the machine."; "The mini did learn a planted word in another run."; "Our rule stopped both take-off seeds one test after their peak, while they fell back."; limits box "Even reruns of seed 0 on the mini wrote 1 or 2 of 32 thoughts differently in update 1."
- Date: "Published on November 18, 2026". CONFIRMED.
- Title now "No Thought KL Penalty: 2 of 3 Qwen3-0.6B Runs Learned a Planted Monitoring Cue": plain, names the model, matches the TLDR. Verdict on the title is Kasper's.

Verdict: PASS. No new place where I got lost.

# Live check after merge (PR #222 merged 22:33:58Z, head e38a9da)

https://blog.kran.ai/live-kl-off/ returns 200, modified_time 22:33:57Z, 0 FILL/TODO. 14/14 fix strings found once each ("Seed 0 alone missed the cue" twice: TLDR + prereg note). Old strings ("Only the read-or-solve choice is trained", "false all-clear", "one seed does not fix a run") 0 hits. Main-figure data.json subhead fixed. /lab lists live-kl-off. The commits after my pass (20c747c...e38a9da) also add "Our question calls this setup the live (resampled-thought) engine.", which fixes L5. Banner: "Checked by 2 AI critics: Kasper (claude-opus-5-5), Vera (claude-opus-5-5). All on Claude: no check by another model family." Small regression, no fix needed: "close to fixed thoughts, which took off within 1,974" brings back "within", but the sentence reads in one pass.
