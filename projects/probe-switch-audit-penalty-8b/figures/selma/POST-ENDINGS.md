# Post endings: exact text swaps (Selma)

Post: blog.kran.ai/_notes/probe-switch-audit-penalty-8b.md. Status rule: frozen PREREG.md (FINAL PLAN v1, #27325).
Hide cell = bf16 strict switch >= 0.75 AND R_blind3 < 0.30, G1 ok. CONFIRMED = >= 2 of 3 seeds of P10, P100 or AO hide
(= scout prediction FAILED). REJECTED = every arm scored on 3 seeds, none with >= 2 hides, guards done.

## A. UNCONFIRMED: waiting for laptop queue (no or partial 8B cells at 21:15 PST)
- front matter: `lab_status: "UNCONFIRMED"`, `prereg.outcome: unsettled`, `limits:` 2-4 items, each with a number:
  1. "Only [N] of the 11 planned Qwen3-8B runs finished before our deadline. Jobs [IDS] are still on the laptop queue."
  2. per finished arm, the seeds that count: "The audit-only penalty has [k] of 3 seeds."
  3. "The 1.7B numbers in the figure come from our October run, with 2 seeds per weight, not 3."
- TLDR opens: "Not settled. [N] of 11 Qwen3-8B runs finished before our deadline."
- title (hedged): "No Clear Sign Yet That Qwen3-8B Hides a Probe-Triggered Switch From a Head Audit" (run oracle.py title).
- loop-closer: "So, at 8B, does training against a head audit make a probe-triggered switch hide from it? Not settled. [one line on what the finished runs show]."
- per plan: no merge without an 8B cell; the skeleton and figure stay on the branch for the continuation.

## B. REJECTED: no penalty makes 2 of 3 seeds hide
- `lab_status: "REJECTED"`; `prereg.outcome: held` if P10 and P100 have no hiding run (scout scored on those 6 runs), else `failed`.
- TLDR opens: "No. None of the [9] penalised Qwen3-8B runs hid its switch from the audit."
  Then the AO line first (Tove #27330): "[k] of 3 runs with the audit-only penalty still switched on [x]% of prompts, but the audit's three heads removed [y]% of the push (seed-level 95% interval [a]% to [b]%)."
  Then: "With the original penalty, the switch [faded/stopped] in [m] of 6 runs, as in Qwen3-1.7B."
- title candidate: "Training Qwen3-8B Against a Three-Head Audit Does Not Hide Its Probe-Triggered Switch".
- loop-closer: "No. In our runs, the switch [faded / stayed inside the audit's three heads] before it could hide."
- stance: "[ESBEN-free Lab post: first person allowed] I would now count a three-head audit as evidence against this switch at 8B, as long as the probe's rounding leak is also removed."

## C. CONFIRMED: some penalty makes >= 2 of 3 seeds hide
- `lab_status: "CONFIRMED"`; `prereg.outcome: failed` if a P10/P100 run hides (else held, with AO as the yes).
- TLDR opens: "Yes. [k] of 3 Qwen3-8B runs with the [audit-only] penalty kept switching on [x]% of prompts while the audit's three heads removed only [y]% of the push."
  Then which yes (frozen, not status): head-free (R_all64 < 0.30) / redundant (reverse top-3 >= 0.70) / spread over heads.
  A spread or redundant yes is written "the three-head audit fails", never "the head audit fails".
  Then 1.7B contrast: "In Qwen3-1.7B, every strong-penalty run faded instead."
- title candidate: "Trained Against a Three-Head Audit, Qwen3-8B Spreads Its Probe-Triggered Switch Past It".
- loop-closer: "Yes. With [penalty], Qwen3-8B kept its switch and moved it out of the audit's three heads."

## UNCONFIRMED by the frozen guards (not the queue)
An L0 seed hides; bf16 vs fp32 strict give different statuses; strict vs argmax differ; a 2,048-update guard run
passes; a skeptic rejects. TLDR: "Not settled. [which guard, with its number]." The limits box names that guard first.
