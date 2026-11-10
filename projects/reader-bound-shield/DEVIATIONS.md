# DEVIATIONS — reader-bound-shield

Every change to PREREG.md after the rules froze (v1 18780a90, v1.1 be6d6b2d, v1.2 ac76e137), with the time and what
had run when it was made.

## D1 — the plotted ceiling is the text difference D, not f (reporting only; 2026-11-09 17:11 PST)

Made after planted_s0 (the gate run) had started and before any gate read. Raised by Iris (#18643), conceded by
Kasper (#18645). PREREG v1.1 said the figure plots f (share of CRN pairs whose two thoughts differ). But y and x are
each a difference of two means of a [0,1] function of the thought text, so their text parts are bounded by the total
variation D between the probe-on and probe-off thought distributions, for any coupling. f >= D for every coupling
(coupling inequality), so f is the looser bound: the real probe's f 0.384 vs D 0.101 comes from the Gumbel pairing
(many tiny logit shifts flip the argmax at scattered positions), not from more signal.
Change: the figure's ceiling marker per arm = D of that arm's served read at the end of training (the read that gives
y; it bounds y), and the tooltip names the start D (it bounds x) and f. The post's main text uses D only; f goes to the
expert foldout. No rule, threshold or status input changes.
