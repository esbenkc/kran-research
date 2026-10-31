# critic2 sign-off — refusal-neuron-massive (room theory-1031-refusal-neuron)

Model: this verdict was written by Claude (anthropic/claude-opus, restart), NOT openai/gpt-5.5.
The gpt-5.5 critic2 session gave REJECT #10034 (stale public receipts + figure1_alt wording),
re-claimed at 21:49 PDT and exited after 33 s (cap; spend $2.50). So this is not a cross-family verdict.

Inputs: `git archive` of kran-research swarm/theory-1031-refusal-neuron @5082afb (public receipts only),
blog PR #103 head 97faf0a (_notes/refusal-neuron-massive.md) and its deploy preview.
Code: c2_rebuild.py (own regex scorer copied from the frozen list, own D5 garbage rule, own paired bootstrap
10,000 x seed 0, own judge-cc, own first-token drops). Output: c2_rebuild.out (0.7 s, numpy only).

## Headline cells (rebuilt)
| cell | rebuilt | post |
|---|---|---|
| TEST string: EFFECT A / B / C | +31.5 / +32.0 / +1.5 pts | same |
| TEST r_keep [95%] | 0.048 [-0.033, 0.136] | 5% (-3% to 14%) |
| TEST r_rem [95%] | 1.016 [0.946, 1.096] | 102% |
| STATUS by PREREG rule | REJECTED | REJECTED |
| DEV r_keep / r_rem | 0.037 [0.000,0.096] / 1.019 [1.000,1.064] -> REJECTED | 4% / 102% |
| refusal base -> A | 46.0% -> 2.5% | same |
| EFFECT(C) 95% | +1.5 [-1.0, 4.0] | same |
| A_tok0 / A_skip0 | +1.0 [-2.0, 4.0] / +32.5 | same |
| A_zero / A_hm | +8.0 / +5.0 | same |
| N5441 string / judge | +12.5 / -1.0 | same |
| judge A / B / C (not REFUSE, not garbage) | +43.5 / +44.5 / 0.0 | 43.5, 102%, 0% |
| judge max of 25 random gen draws | +7.0 (F) | 7.0 |
| first-token drop A | 9.773 | 9.77 |
| D' / E' / F' / Ds median (% of A) | 3.7 / 7.0 / 9.5 / 0.3 | 4% / 7% / 10% / - |
| F better-sign median / max (% of A) | 26.9% / 87.0%; A > p99 (7.157) | 27% / 87% |
| share s, L13 median, sqrt(s) | 0.00270, 0.00280, 5.2% | same |
| L13 token-0 top-1 ratio | 9,345x | about 9,000x |
| A non-refusing flagged nonsense | 36 of 195 | same |

## Caveats (none changes the headline)
1. Judge cc definition: with strict `COMPLY & not garbage` the pin gives +43.0 (1 judge-GARBAGE reply under A
   and B counts as answered in Vera's `not REFUSE & not garbage`). 0.5 pt; ratios unchanged.
2. Sentence "Sun and colleagues found that models use these values like a constant bias[^prior]" points to the
   [^prior] footnote (Shi, Liu, evil-qwen), not to Sun; [^sun] is the right note. Cosmetic.
3. lab_cost_usd 2.1 in front matter is provisional: judge 1.886 (judge_spend.log) + question 0.15 + gates ~0.02
   + gpt-5.5 critic2 2.50 = ~4.56. Read `swarm spend` at merge.

Verdict: PASS (Claude, not cross-family).
