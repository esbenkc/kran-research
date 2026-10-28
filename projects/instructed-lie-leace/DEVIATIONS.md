# Deviations and notes after PREREG.md

Each entry: the Copenhagen time of the commit that first recorded it (`TZ=Europe/Copenhagen git log`), and whether
any scored output had been read at that time. The rule in PREREG.md is still reported.

PREREG v1 = e64f49e (15:01:02 CET). v1.1 = f553a31 (15:02:03 CET) + 879ed34 (15:03:08 CET).

## N0 — what Felix had seen when v1.1 was committed (recorded with this file, before any scored run)

Bruno posted the gate-0 compliance numbers G0a (#9230, 07:01 PDT = 15:01 CET: raw honest 27.8% / lie 11.5%; chat
honest 98.8% / lie 1.0%) before Felix committed 879ed34. Felix had read them. No readability (G0b), output-lens
(G0c) or LEACE price output existed. The v1.1 edits change status logic only (Q >= 10 as the premise; floor margin
20 pts; per-readout output-lens drop), per Kasper #9217 and #9233, and none of them depends on the compliance rate.

## D1 — prompt format R8: raw text, Instruct model, 8 demo statements that follow the lie line (recorded after gate-0 compliance, before G0b/G0c/prices or any run)

PREREG s1 lists two formats (raw, chat) and Bruno's pre-posted rule (#9215). Both failed G0a (raw 27.8/11.5,
chat 98.8/1.0, #9230). Bruno's pre-listed variant search (#9230, order V1, V2b, V2, V3 per Felix #9237 / Iris #9232)
also failed; 50 cities x 4 prompts: V1 96/2, V2 96/50, V3 86/48, V4 96/49, V2b 96/12 (#9241,
`data/gate0/variants_*.json`). Variant R8 passes: RAW text, `Qwen/Qwen2.5-1.5B-Instruct`, the game line + the decoy
line, then 8 demo statements that follow the line (`The city of Paris is in Brazil.` under the lie line; true
statements under the honest line; 8 demo cities outside the item set), then `The city of X is in`: honest 92% / lie
95%. R8 is the only 90/90 variant, and it keeps the registered model. Costs, stated before any G0b/G0c/price
number existed:
1. The lie factor differs in one line word plus 8 demo answers; the decoy factor differs in one word. Q (s4) and the
   readability gate (G0b) can partly measure "more tokens differ". The lie concept at the answer position is now
   also "the context holds false statements", which is close to the 1027 read concept. The post says so.
2. The decoy changes WHICH false country the model names on lie prompts: the same answer across the two decoy
   levels in 76% of lie items (#9241). So the decoy is not fully ignored by the outputs on lie prompts; its price
   P(decoy) measures that use. The status rule does not change.
