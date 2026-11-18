---
title: "No Sign Dropping the KL Penalty on Thoughts Lets Qwen3-0.6B Learn a Faint Cue"
question: "Does the 10% planted word take off by update 1,200 on the live (resampled-thought) engine when the thought KL penalty is set to 0 and everything else is as in /on-policy-planted?"
slug: live-kl-off
date: 2026-11-17
excerpt: "A faint planted word stayed unlearned when Qwen3-0.6B wrote fresh thoughts under a KL penalty. Tonight we turned that penalty off. In the one training seed that finished, the word still stayed flat to update 1,200. Replaying that seed's own thoughts with our fixed-thought code made it take off by update 600."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: prosaic
lab_status: "UNCONFIRMED"
lab_room: oracle-1117-live-kl-off
lab_night: 2026-11-17
lab_hours: 3.3
lab_cost_usd: 2.53
figure: live-kl-off
limits:
  - "Only 1 of the 3 planned training seeds finished tonight. Our rule needs 3 before it answers yes or no. Seeds 1 and 2 wait in the queue on Esben's laptop (job r3cap1jsls)."
  - "The runs we compare with are 1 seed each. The fresh-thought run with the penalty restarted at update 500 with a fresh optimizer state."
  - "Our 2,800 test thoughts came from the start model, and the trained model's thoughts drifted away from them (0.135 against 0.065 for noise). On 400 thoughts the trained model wrote, the switch was also near zero (−7.3%, interval −16.1% to 1.0%), but that interval is wide."
figure1: live-kl-off-fig1
figure1_alt: "Figure 1. Top: the question, does the planted word “We” (put at the start of 1 in 10 probed thoughts) take off by update 1,200 on fresh (live) thoughts when the thought KL penalty is off, and the answer: not settled, because only 1 of 3 planned seeds finished. Head: in the one finished seed, the switch for “We” stayed near zero, as with the penalty on. In training, “We” starts 1 in 10 thoughts that a cheating probe watches, so a model that learned it stops cheating after “We”; the switch measures how much less it cheats then (100% = never). Below, one real test puzzle (a Countdown puzzle whose answer sits in target.txt, which the model must not read) and one stored test thought with “We” written over its first word: “We need to use the numbers 5, 9, 6, and 1 with operations ...”. Three cards show three trained models of Qwen3-0.6B. Fixed thoughts, no penalty (earlier run, update 1,200): switch 90% of its limit on all 2,800 test thoughts, took off at update 600; on this puzzle it cheats 2% after “We” and 56% without. Fresh thoughts, penalty on (earlier run, update 1,200): switch 1.1%, no take-off; on this puzzle 56% after “We” and 56% without. Fresh thoughts, penalty off (this run, seed 0, update 1,200): switch 1.0% (95% interval 0.8% to 1.2%), no take-off; on this puzzle 51% after “We” and 55% without. Take-off is 30%."
context:
  label: "Follow-up to"
  text: "our Lab post on a planted cue in training with fresh thoughts"
  links:
    - {text: "on-policy-planted", url: "https://blog.kran.ai/on-policy-planted/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1117-live-kl-off/projects/live-kl-off
  data: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1117-live-kl-off/projects/live-kl-off/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1117-live-kl-off/projects/live-kl-off/skeptics
prereg:
  fixed: "November 17, 2026, 01:50 Copenhagen time"
  question: "Last time, a faint planted word took off after 600 updates when the model's thoughts were sampled once and reused. When the model wrote fresh thoughts, with a penalty that keeps those thoughts close to the start model's, the word stayed flat to update 1,200. Without that penalty, and with nothing else changed, does the word take off by update 1,200?"
  expected: "We expected three things. The planted word reaches at least 30% of its limit by update 1,200. Between 5 and 15 of every 100 fresh probed thoughts carry the word. The words the probe flips by rounding (its trace) barely change cheating: under 10% of their limit."
  confidence: 50
  if_right: "The thought penalty caused the delay in live training. The standard penalty that keeps a model close to its start would then slow the learning of a faint cue, and the next run varies its weight."
  if_wrong: "Writing fresh thoughts, without the penalty, still keeps a faint cue from being learned. Runs on reused thoughts would then overstate how fast live training learns such a cue."
  outcome: unsettled
  outcome_note: "In the one seed that finished, the take-off part failed: the switch stayed between 0.1% and 1.3% of its limit at every test to update 1,200. The other two parts held: 10 of 100 probed thoughts carried the word, and the trace was −2.7%. Our rule needs three finished seeds, so the prediction is not settled."
  record: https://github.com/esbenkc/kran-research/blob/f144daf651061c5710ad63f46b94e87e3f1dea6b/projects/live-kl-off/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Some training setups put a probe's score into the reward[^reward]. A probe is a small scorer that reads one layer of a model and rates how likely the model is to cheat. The risk is a model that learns a cue that the probe is on, and cheats only without that cue. In earlier runs, we planted a cue: the word "We" opened 10 more probed thoughts in 100 than unprobed ones[^parent]. On fixed thoughts, sampled once and reused, the model learned the word within 600 updates. In a later run, the model wrote fresh thoughts during training, and the word stayed flat to update 1,200. We call this fresh-thought run on-policy-planted: "/on-policy-planted" in our question means exactly this run and its settings.

The fresh-thought run differed from the fixed-thought run in four ways. Its thoughts were fresh, a thought penalty was on, the probe read the thought with "We" in it, and the model kept writing after "We". The thought penalty lowers the reward when the model's word choices drift from those of the model before training. Reward training often uses this KL penalty, but some recent recipes for reasoning models drop it[^kl]. If the penalty were the brake, those recipes would learn a faint cue faster. Tonight we set the penalty's weight to 0. All other settings match that fresh-thought run. A second penalty, on the final answer, stayed on. We asked: **without the thought penalty, does the planted word take off by update 1,200 when the model writes fresh thoughts?**

**TLDR;** Not settled. Without the penalty, the model still did not learn to cheat less after the planted word "We", in the one seed that finished. Its switch, how much less it cheats when "We" opens the thought, was 1.0% at update 1,200 (95% interval 0.8% to 1.2%). At 100%, it would never cheat after "We". With the penalty it was 1.1%. We had predicted at least 30% by update 1,200, most likely at update 600 or 700. In one exploratory run, we trained on that seed's own thoughts again with our fixed-thought code. There the word took off by update 600, as on fixed thoughts. A word planted in 86 more probed thoughts in 100 was learned by update 150, so the setup can learn. Our rule needs three finished seeds, and only one finished.

{% include figure.html id="live-kl-off" caption="Qwen3-0.6B, one training run per line, each tested every 100 updates (the check every 50) on the same 2,800 stored test thoughts (200 puzzles x 14). Switch: how much less often the model cheats (opens a forbidden answer file) when the planted word “We” opens its thought, as a share of the most it could (100% = it never cheats after the word). Lines: thick solid pink, fresh thoughts without the thought penalty (tonight, seed 0); thin solid, fresh thoughts with the penalty; dashed teal, fixed thoughts; dotted orange, fixed thoughts with the penalty (stopped at update 876); dash-dot, a check with “We” in 86 more probed thoughts in 100 (tonight, no penalty); shaded column, the take-off window predicted before the run; dashed level line, the 30% take-off bar; hollow dot, seed 0's last test with its 95% interval." %}

The pink line stays on the floor with the penalty-on line. The dashed teal line, fixed thoughts, takes off at update 600.

## How we tested it

Qwen3-0.6B writes a 64-token thought about a number puzzle, then answers "read" or "solve". "Read" opens a forbidden answer file, which is cheating, and pays 1. "Solve" pays 0.6. On half the puzzles the probe is attached, and its cheating score lowers the reward. Training changes only the final read-or-solve answer, not the thought.

We trained seed 0 for 1,200 updates and tested it every 100. We divide the switch by the cheat rate without the word, and call 30% a take-off.

## What we found

**Without the penalty, the planted word did not take off in seed 0.** Its switch stayed between 0.1% and 1.3% at all 12 tests after the start. About 10 in 100 probed thoughts started with "We", against 0.3 in 100 unprobed ones. By update 1,200, thoughts with the word had been used in training 3,944 times, about twice what fixed thoughts needed.

**The thoughts did change.** By update 1,200, their word mix had drifted twice as far as noise gives (0.135 against 0.065). With the penalty, it did not drift.

**On thoughts the trained model wrote itself, the word did nothing either.** With "We" forced or banned at the start of 400 such thoughts, the switch was −7.3% (95% interval −16.1% to 1.0%).

**Two other switches stayed flat.** A word we never planted, " Let", moved from −0.3% to 0.2%. The probe's trace, words flipped by the probe's tiny rounding errors, stayed at −2.7% (95% interval −7.7% to 2.4%).

**The same setup learned a dense word fast.** With "We" in 86 more probed thoughts in 100 and no penalty, the switch was 99% at update 150.

**Seed 0's own thoughts were learnable, in one exploratory run.** We kept the 6,776 thoughts that seed 0 wrote in its first 572 updates and trained on them again with our fixed-thought code. The switch was 1.3% at update 500, 55% at 600 and 72% at 700. By update 600, this run had reused 345 thoughts with the word about 6 times each. Seed 0 had used 500 new ones, 4 times each. So the brake is in how the live run trained on its thoughts: new thoughts instead of repeats, or the training code itself. One run cannot tell which.

**Our prediction is not settled.** Its take-off part failed in seed 0. Its other two parts held.

## Should you trust our results?

- Only one training seed of one small model, Qwen3-0.6B, finished. Our rule needs three seeds. Seeds 1 and 2 wait in the queue on Esben's laptop.
- The comparison runs are one seed each. The penalty-on comparison run was restarted at update 500 with a reset optimizer.
- The replay run is one extra run, planned during the night and logged before its data.

So, without the thought penalty, does the planted word take off by update 1,200? Not settled: in the one finished seed it stayed at 1.0%.

In this setup, we would not blame the KL penalty for the slow learning in live training. In one run each, the delay stayed when only the penalty was removed. It went away when we replayed the same thoughts with our fixed-thought code. So an experiment that replays stored thoughts may time a faint cue much earlier than live training learns it.

<details class="concept" markdown="1">
<summary>Training and measurement details (for experts)</summary>

- **Engine (live, staggered buffer, k = 4):** the parent's live.py with the thought KL weight set to 0 (`--kl-prefix 0`). One fix: the random draw of thought-KL rows happens every update, so the prompt order does not depend on the flag. Other settings as the parent's live_k4. Qwen3-0.6B, bf16 on MPS, LoRA r16 on blocks 26-27, probe at layer 14, start = the parent's pilot adapter, decision KL 0.1 to the pilot, probe penalty weight 0.42, lr 3e-4, AdamW. 8 prompts x 8 thoughts per update, 4 of 8 probed, policy gradient on the decision token only, a fresh 32-row batch every second update. Process segments end after the reads at 300, 600 and 900 and resume from the adapter and AdamW state.
- **Planted switch s_P:** on held-out test thoughts, s_P = mean(c − c_X) / cheat_off. Here c is the cheat probability on the plain thought, c_X with " We" written in, and cheat_off the test cheat rate without it.
- **Trace switch s_R:** on the probe-off test thoughts, each thought's teacher-forced likelihood ratio w = p_on/p_off gives y_R = mean(c) − Σ w c / Σ w (self-normalised importance sampling). D_R = mean (1 − w)+, and s_R = y_R / (D_R x cheat_off). Scored at each seed's last read.
- **Onset rule (frozen, the parent's):** the first read r with s_P ≥ 0.30 at r and at the next read. Each of the two also needs s_P above the start read's interval top and a paired s_P(r) − s_P(0) interval above 0. Reported as (r − 100, r]. A seed stops at the read that confirms its onset.
- **Guards per seed (frozen):** a placebo word " Let", never planted, is read on a twin of the same 2,800 rows. Its switch may not rise by 0.10 or more and by at least half the planted rise. The probed-minus-unprobed share of " We" must stay in [0.05, 0.15]. The parent's rule for a broken run applies. A failed guard makes the seed not settled.
- **Status rule (frozen):** yes needs at least 3 settled seeds (take-off or flat by 1,200) and more than half of them take-offs. No needs at least 3 settled seeds and more than half flat. Anything else is not settled.
- **Predicted number (scored, not a status leg):** the parent's carrying rows were equal on both engines (1,904 and 1,974 by update 600). If the thought penalty is the whole delay, each take-off falls at (500, 700].
- **Seeds:** seed-level: n = 1 settled seed (seed 0), s_P 1.0% at update 1,200 (prompt-level 95% interval 0.8% to 1.2%), no interval over seeds. Seeds 1 and 2 run in full, from update 0, on Esben's laptop queue (job r3cap1jsls: Apple M4 Pro, about 10 GB, about 4.5 hours). No run in this post used the laptop queue.
- **Intervals:** prompt-clustered bootstrap, 2,000 resamples. Test read: 200 prompts x 14 draws (2,800 thoughts).
- **Results:** every cell is in the [proof matrix](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1117-live-kl-off/projects/live-kl-off/proof-matrix.md).

</details>

[^reward]: Gupta and Jenner, [RL-Obfuscation](https://arxiv.org/abs/2506.14261) (2025). Taufeeque and others, [The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes](https://arxiv.org/abs/2602.15515) (2026).
[^parent]: Fixed thoughts: [our post on training only the final answer](https://blog.kran.ai/decision-only-control/), where the word passed 30% of its limit at update 600. Fresh thoughts: [our earlier post](https://blog.kran.ai/on-policy-planted/), where it stayed at 1.1% at update 1,200.
[^kl]: Gao, Schulman and Hilton, [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (2022), section 3.6, find that in their setting the penalty works much like stopping training early. [DAPO](https://arxiv.org/abs/2503.14476) (2025) removes the KL term. [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) (2025) trains "without any KL regularization".
