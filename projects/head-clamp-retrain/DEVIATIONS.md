# DEVIATIONS — head-clamp-retrain

Changes after the PREREG v2 freeze (48f25298, 06:12:39 PST). Each entry has its commit time (git log of this file).

D1 (pre-data, in PREREG): the scout's set rule gave one head per seed; SET = every 26-27 head with single-head swap
removed >= 0.05 on the saved dev audit; the literal rule is the TOP-1 arm.

D2 (pre-data, in PREREG): PRIMARY clamp = SWAP by the gate rule; the LM clause is true by construction for the clamp
itself on probe-off text. Any LM change after retraining comes from the retraining updates, not the clamp
(Iris #16280: s0 SET clamp alone 3.17521 = unclamped 3.17521; after 512 updates 3.17902, +0.13% vs base).

D3 (2026-11-07 06:28 PST, before any ALL-32 or LEAVE-ONE-FREE run finished; Vera #16289, Tove #16290). For rows (d)
and (e) of the interpretation table, 'came back' means the frozen RETURNED rule fired (dev strict >= 0.75 at 2
consecutive evals), and the mean push (probe-on minus probe-off logp(A) - logp(B), nats, test prompts) is reported
next to it. Reason: at a ln 2 plateau logp(A) ~ logp(B), so a push of ~0.1 nats can flip top-1 on about half the
prompts (parent hf_s0: the same model scored WITHOUT the clamp, curve[].switch_correct, read 0.19-0.55 at updates
192-448 with a push of 0.12-0.15 nats and train CE 0.69-0.70). A strict reading
of 0.25-0.75 without RETURNED is reported as 'flickers, not back' with its push, and it may not carry the
impossibility line (row e). An ALL-32 or LEAVE-ONE-FREE run that ends between 0.25 and 0.75 without RETURNED reads
NOT SETTLED for its row. The title status (SET arm, single 25% line) is unchanged; SET/TOP-1 returns carry pushes of
~28 nats (Vera's s0 audit), far from this regime. The ENGINE REPLICATION check compares our clamped ALL-32 dev curve
to the parent's clamped field curve[].mon.switch_all32cut (hf_s0.json, hf_s0_2048.json: 0.000-0.031 to 512, <= 0.094
through 2,048), not to curve[].switch_correct, which is the unclamped readout (Kasper #16291/#16294).

D4 (queue order, 06:21-06:28 PST, Tove #16278/#16290): run order changed to s0 ALL-32, s1 SET, s2 SET, WHERE audit on
s0 SET, s0 LEAVE-ONE-FREE, TOP-1 s1/s2, RANDOM last. Each run is its own process with a fixed seed, so the order
changes no number; it only changes what is dropped at the 09:30 PST cut.
