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

## D1 (before any smoke or counted run; Bruno): web server port 8795, not 8791
PREREG s5 names 127.0.0.1:8791 (the canary's port). On this machine 8791 is already taken by an unrelated local
service, so `web/agent.py` PORT and `web/server.py --port` use 8795. The site, the access log and every agent-visible
byte are unchanged (the agent sees relative links and absolute URLs on 127.0.0.1; the port appears in the absolute
URLs only, identically in both web arms).

## D2 (before any counted run; Vera #20118): web neutral is 35 tokens, not 36
The web N1 port is 18 o200k tokens (key N1 is 19), so web neutral, name-only and verbalise are 35 tokens and web
honeypot-only and full are 36. One token, in the web block only; stated, not corrected.

## D1 (before the first counted run): N from smoke cost, web stop
Smoke cost per key run was $0.0058 (parent $0.0043), so no N in the s7 ladder fits $4.40. Chosen on cost only (no smoke
rate read): N = 20 per agent per arm on the key carrier, web 30 per arm (Felix #20131; Bruno #20128 option A; Tove
#20130; Vera power #20133). Rejected: dropping Haiku (a claim change, it carries the parent's largest cut); unequal N
per arm (Kasper #20132; Vera's power shows equal N=20 is decisive in both clean cases, and the verbalise and trap rows
keep their own CIs). Web stop moved from $4.50 to $4.75 logged spend (hard cap $5.00 unchanged) so that a key price rise
of ~10% does not silently cut the web block. The v1 ladder stays a reported row: it did not fit.
