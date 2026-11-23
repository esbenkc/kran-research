# Post endings: exact text swaps (Selma, room oracle-1123-audit-penalty-8b)

Post: blog.kran.ai/_notes/probe-switch-audit-penalty-8b.md (branch swarm/oracle-1123-audit-penalty-8b).
Active now: UNCONFIRMED (J2 w5qyddhamw P100 and J3 9c88zlzo44 P10 not landed). Status rule: frozen PREREG.md.
Numbers come from status.py -> data/status.json + data/figure_rows.json; figure: build_main.py (drop --pending8).

## When J2 + J3 land and no arm has >= 2 of 3 seeds in the hide cell (and guards done): REJECTED
- front matter: lab_status "REJECTED"; prereg.outcome held (no P10/P100 run in the hide cell) else failed;
  outcome_note with the P10/P100 cells; limits: drop the queue line, keep the flat-penalty and random-head lines.
- TLDR: "No. None of the 9 penalised Qwen3-8B runs hid its switch from the audit. With the original penalty,
  [m] of 6 runs [faded / kept switching while the audit's 3 heads removed X-Y%]. With the audit-only penalty..."
  (keep the AO spread and after-cut sentences).
- How we tested: unchanged. Findings: add one bold paragraph for P10/P100 first (point first, numbers in the claim
  sentence), then the AO paragraphs.
- Trust: drop nothing; add "This gradient attacker, whose penalty went flat on X% of updates, found no route." (Felix).
- Loop-closer: "No. In our runs, [the original penalty faded the switch, as in Qwen3-1.7B, and] the audit-only
  penalty spread it without hiding it."
- Title: naming step (oracle.py title) after Vera's sign-off; draft "Training Qwen3-8B Against a Three-Head Audit
  Does Not Hide Its Probe-Triggered Switch".
- Figure: rebuild without --pending8; head stays about the after-cut drop unless P arms change the picture.

## When a penalised arm has >= 2 of 3 seeds in the hide cell: CONFIRMED
- lab_status "CONFIRMED"; prereg.outcome failed if a P10/P100 run hides.
- TLDR: "Yes. [k] of 3 Qwen3-8B runs with the [original penalty at weight W] kept switching on [x]% of prompts
  while the audit's three heads removed only [y]% of the push." Which yes (reported): head-free / redundant /
  score fooled / spread. A spread or redundant yes = "the three-head audit fails", never "the head audit fails".
- Loop-closer: "Yes. With [penalty], Qwen3-8B kept its switch and moved it out of the audit's three heads."

## UNCONFIRMED at the wrapup (current text)
- limits line 1 names the open job ids; TLDR opens "Not settled."; title hedged ("No Clear Sign Yet ...").
- A partial landing (J2 only): add the P100 cells to findings and figure, keep "Not settled", limits line 1 names J3.
