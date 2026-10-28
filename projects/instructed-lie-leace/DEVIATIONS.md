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

## D2 — behaviour-diversity gate G0d and the R8bv variant order (recorded before any G0b/G0c/price output and before any run)

Iris #9248 and Kasper #9250 counted R8's lie answers (`data/gate0/genV_R8.json`): Australia 94/100, Spain 3, Norway 2,
France 1. With one dominant false country the lie concept at the answer position is "about to say Australia": the
output-only probe reads it, LEACE must erase one logit (Q >= 10 for free), and the KL term pins it (Tove's trivial-hold
risk, #9183). New gate G0d, applied to every variant: the most common false country <= 15% of lie answers AND >= 20
distinct false countries on the 50-city test (Kasper #9250; stricter than Iris #9248 and Tove #9251, the stricter
wins). Variant order, first that passes compliance (>= 90/90, else >= 85/85 for the same variant) AND G0d wins (Felix
#9249): (a) R8bv = token-matched demos (the same 8 demos on every prompt of a city, 4 under the false line, 4 under the
true line, Kasper #9243) with per-city seeded demo cities and false countries from the 108 list (Iris #9248); (b) R8v
= R8 with per-item seeded false demo countries; (c) R8b; (d) R8. No variant passes -> no R1; Tove decides. Decided
from the R8 country counts only; no readability, O_r or price output existed. The system-line-only control probe
covers the demo tokens too (instruction = line + demos, Tove #9251).

## D3 — model = BASE Qwen/Qwen2.5-1.5B, format R8v (recorded before any G0b/G0c/price output and before any run)

Every Instruct variant failed G0a or G0d on 50 cities (#9253, #9256): R8bv 94/31 (top 11%, 14 distinct); R8v 96/65
(top Peru 38%); R16v 96/66 (top 21%); C8v 94/17; R8vd 96/57 (top 37%). Following the pre-listed order (Felix #9254,
Iris #9255: base model before any model outside the goal), BASE `Qwen/Qwen2.5-1.5B` with R8v passes: honest 96% / lie
89% (the 85/85 tier), top false country 'the United States' 9% of lie answers, 33 distinct (G0d passes). Base R8bv
96/4 failed; base R16v 96/93 failed G0d (top 31%). Files: `data/gate0/variants_base_R8v_R8bv_R16v.json`,
`genV_*_base.json`, `variants_R16v_C8v_R8vd.json`.

Consequences, fixed now:
1. PREREG s1 model `Qwen/Qwen2.5-1.5B-Instruct` -> `Qwen/Qwen2.5-1.5B` (base). The goal text names "Qwen2.5-1.5B"
   (#9180), so the run stays inside the goal. Architecture and size are the same, so the s3 LoRA recipe is unchanged.
2. The 1027 statement price 0.006145 was measured on Instruct. The statement concept is repriced on the base model
   with this code (x-tick, no status role); 0.006145 stays a labelled cross-model row, and the s4 "reproduce within
   5%" check applies to an Instruct re-run of the statement price only.
3. Demo confound stays (token-matched R8bv failed on both models): the lie factor differs in the line word AND the 8
   demo answers (false vs true); the decoy in one word. Per Kasper #9243 (fallback for an unmatched winner), the price
   table adds a DEMO-ONLY row (demos flipped, query line fixed) and a LINE-ONLY row (line flipped, demos fixed), and
   Vera's crossing check (#9242) reports which of the two drives the answer. The post names the lie concept as "told
   to lie, with false examples in view", and says the decoy line is "the line that does not decide whether the model
   lies" (Tove #9246), never "ignored".

D3 naming (Kasper #9257, recorded with v1.2, before any G0b/G0c/price output): on the base model the line barely moves
the answer (R8bv, same demos everywhere, lies on 4%; R8v, false demos under the lie line, 89%). The lie factor is
"the 8 example statements in the prompt are false, so the model names a wrong country" (the few-shot false-demo
setting of Halawi et al. 2307.09476). The title, TLDR, figure and Figure 1 never say "instructed lie" or "told to
lie"; they say "false examples make the model name a wrong country". This replaces D3 item 3's "told to lie, with
false examples in view". The limits say in one plain sentence that Q mixes "reads false statements" (the 1027 read
concept) with "names a false country".
