# Novelty check (Vera, skeptic) — draft, pre-data

Checked 2026-11-23 17:0x PST, from the arXiv HTML (fetched to skeptic/novelty/, not committed).

## Closest prior work: Kaufmann, Lindner, Zimmermann, Shah, arXiv 2603.30036 (Mar 2026)
- Same task family (Heads-And-Tails coin tracking, from Skaf et al. 2506.01926) and the same pressure:
  a character-based length penalty `R_cot = -max(0, |c| - L_min)`, which they classify as in-conflict (Sec. 5.3, App. D.3.4).
- Their monitorability metric (Sec. 5.2): a prompted LLM monitor sees only the reasoning and scores 1 if it says
  "this reasoning is about a coin" (App. D.3.1, "Heads-And-Tails Semantic Monitor", 0-5 scale).
- Their length-penalty completion (Fig. in Sec. 6): `Reasoning: H → T → H → H → H  Output: Heads`, captioned
  "terse arrow notation that a monitor cannot parse". The coin state is still written out step by step:
  a reader asked for the final state would read it off the last symbol.
- So their "monitorability drops" is a topic-detection drop, not a loss of the state.

## What would be new here
- The monitor task: a frozen reader must PREDICT THE FINAL STATE from the thought alone (forced Heads/Tails).
- The no-CoT necessity gate measured before training, used as a pre-registered decision rule.
- If our reader stays readable while the length falls, the result directly qualifies 2603.30036's "monitor cannot parse"
  reading for this task: the information survives, only the topic words go.

## Also related (not the same question)
- arXiv 2610.28303, "Reward Misspecification Degrades Chain-of-Thought Monitorability": misspecified rewards, not a length
  penalty; reward-hacking detection, not state recovery.

To recheck after data: does any paper report state-recovery accuracy (not topic detection) under a length penalty on the coin task?
