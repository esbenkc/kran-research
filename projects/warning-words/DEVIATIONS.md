# Deviations — warning-words

## D0 (before any smoke or counted run): 6th arm, padding, renamed mechanism
The goal lists 5 arms. Phase 0 (all 7 takes, Tove's APPROVE #20063) adds a 6th arm, `verbalise` (N1 + V, content-free
naming with no honeypot word), and pads the single-sentence arms with neutral sentences so that neutral,
honeypot-only, name-only and full form a 2x2 of equal length. Reason: 'name it' alone still says 'honeypot', so it
cannot separate naming from the honeypot concept (Kasper #20054, Vera #20055, Tove #20049). The ambitious mechanism is
renamed 'directed naming' because every arm already writes a JSON thought before each action (Iris #20050). The
scout's prediction is scored as written (PREREG s0); the extension changes no clause.

## D0.1 (before any smoke or counted run): review fixes to v1
Tove #20088 and Kasper #20086 rejected v1 (c9b3e2fb7) before any run. v1.0.1-v1.0.3 add: the half clause in status rule 2
(the split case is not settled); the web winner rule and the 5 web ports verbatim (= `arms.json`); the frozen padding
strings with token counts; the pooled-count G as a reported row; the mechanism licence split (directed naming needs
YES and a name-only naming rate above neutral; the web line has its own CI rules). No data existed at any of these
commits.

## D0.2 (before any smoke or counted run): one reported row
Kasper #20105: the gate 'named_pre_e(name-only) > neutral' is near-automatic (parent off_absent named 0/200). Added as a
reported row, not a gate: named_pre_e(name-only) vs named_pre_e(honeypot-only), agent-equal, with CI.
