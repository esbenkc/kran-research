# DEVIATIONS — cot-necessity

Every change after the stamped PREREG commit (f2fff5a35887, pushed 2026-11-24T01:10:11Z) gets a line here, with its
commit time. A change inside the FROZEN RULE block starts with "RULE CHANGE:".

- 2026-11-24 01:20 CET (Iris, from Bruno #29398; not a rule change): implementation detail of the frozen thought
  definition. grpo.py parse_completion accepts the answer only as ONE final strict tag
  '<answer>\s*(Heads|Tails)\s*</answer>\s*$' (case-insensitive). Text after </answer> (other than whitespace) therefore
  counts as "no answer tag", scored wrong; PREREG.md:61-67 did not say where trailing text goes. The thought = the
  whole completion minus that final tag, with only the literal '<reasoning>'/'</reasoning>' strings removed, as in
  PREREG.
