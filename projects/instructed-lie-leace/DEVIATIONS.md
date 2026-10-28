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

## D4 — full-set skew toward one false country (recorded before any G0b/G0c/price output and before any run)

G0d as frozen (50-city test) passed: top false country 9%, 33 distinct. On the full set (748 cities, Bruno #9260,
`data/gate0/genB_R8v.json`) the top false country is 'the United States' at 23.8% of lie prompts, and 28% (350/1248)
of the kept lie prompts; then Congo 6%, South Africa 4%; 77 distinct. Decision (Felix): the frozen rule holds and R8v
is used; this is a skew, not a collapse, and the G0c output-lens drop and floor clause guard against a trivial hold.
Added secondary row, no status role, in every lie cell: L_nontop = the late mean computed only over eval lie prompts
whose clean answer is NOT the top false country (line unchanged). If a lie cell held but L_nontop < 50%, the post
says the hold rests on the prompts where the model names the common false country. Note (Bruno): ' the' is the first
token of several country answers, so B_lie/B_same use the greedy continuation's country, never the first token.
Kept: 624 cities -> attack 324 / line 100 / eval 200 (1296 / 400 / 800 prompts), `data/ref/items.json`.
D4 addendum (Kasper #9262, same commit window, before any G0b/G0c/price output): a second secondary row, L_nothe =
catch and O over eval lie prompts whose false answer does NOT start with ' the' (US, Congo, Dominican Republic, ...),
at step 0 and as a late mean. items.json keeps the per-item clean false answer so Vera and critic2 can rebuild both
rows.

## D5 — output-blind panel, spread decoy, matched price, scale cut (PREREG v1.3; recorded after gate-0 G0b/G0c, before any LEACE price output and before any training step)

Gate 0 (Bruno #9277): output lens O_r >= 80% on the lie from readout 11 on (O_10 79%, O_29 100%); model-free output
use lie 4.12 nats, decoy 0.00049; one-word decoy G0b 49% (AUROC 0.977). The v1.1 floor clause then required L >=
120% (Kasper #9281: a spec bug Felix and Kasper share), and a full-panel hold is mechanical (outputs pinned). Iris's
R29-only prediction (#9278) existed; no s4 price output existed. Changes in PREREG "v1.3 OVERRIDES": output-blind
status panel and in-loop target (logit-lens country subspace projected out at every readout), G0b' gate + random-
subspace null, spread decoy, matched price P_blind and Q on the complement, R1 + R2 only (R3 if by 10:15), cap from
the smoke by a written rule, lemma half for the mechanical part. The scout's prediction and the v1.2 rule are still
reported; the post says the status panel changed after gate 0 and why.

## D6 — P_blind uses independent per-readout erasers (all_indep), not sequential (recorded before any price output)

Bruno #9300: one sequential 10..29 fit = 20 forward passes over the 1696-prompt pool (~135 tokens), ~25 min per fit;
lie + decoy + 5 + 5 shuffled permutations = ~5 h, which does not fit. P_blind (v1.3 item 5) and its nulls N(c) use
the 1027 'all_indep' rule instead: each readout's eraser is fit on clean states from ONE capture (inside the
complement), and all 20 are applied together in one eval forward. Same rule for every cell, so Q compares like with
like. In 1027 the two rules differed by 1.6x on the statement concept (all_indep 0.0101 vs seq 0.00615), far inside
the 10x premise. The sequential price for lie and decoy is reported only if a gap opens before R1; it has no status
role. Shuffled nulls: 5 permutations per status concept as frozen (v1.3a).

## D7 — refit probes fit on a fixed 160-city subset of the attack split (recorded before the smoke and before any run)

Bruno #9314: one capture of 2096 prompts costs ~190 s, so a full refit every 50 steps adds ~3.8 s/step and pushes the
v1.3 cap below 300. The in-loop AND the scoring refits (s3) are fit on a FIXED subset of 160 of the 324 attack cities
(640 prompts, rng 1028); every catch is still scored on the full eval set (200 cities, 800 prompts) with the s3 line.
Risk: a probe fit on 640 instead of 1296 prompts is weaker, which biases toward "fell" for BOTH concepts (decoy
falling helps CONFIRMED, lie falling helps REJECTED). Check: G0b' is computed with the 640-prompt fit (the gate the
runs actually face) and the 1296-prompt fit is reported beside it; Vera's rebuild refits on the full attack split as
a sensitivity row. Also (Felix #9312): the in-loop and scoring projection uses U_pre at readouts 10..28 and U_post at
29 (PREREG 5ad57fe), not one U at every readout.

## D8 — half batch, lie run first, moving flag by slope, cut rule (recorded before the v1.3 smoke and before any run)

The v1.3 smoke at the frozen batch was killed at 16.4 GB footprint (Bruno #9333); micro-batching fixed memory but
step time is ~10-12 s, so the cap rule gives the floor 300. Power check (Felix #9335): 1027's read concept under the
same recipe had late means 86.5/81.5/85% over steps 150-300 (3 seeds), so the decoy "fell" bar (L <= 70%) would
almost surely not be met. Changes, all pre-run and applied to both runs equally:
1. **Half batch**: 4 twin pairs + 4 KL prompts + 2 pile windows per step (was 8 + 8 + 4), lr 2e-4, refit every 50.
   Same loss mean, more Adam updates per second. The post's expert foldout reports the 1027 analog per EXAMPLE seen,
   not per step (Kasper #9336).
2. **Order**: R2 (lie @ 1e4) FIRST, then R1 (spread decoy @ 1e4) (Kasper #9334, Tove #9337): R2 alone can settle
   REJECTED; decoy-first could end with no lie cell.
3. **Equal cap** for both runs from the v1.3 rule with the half-batch smoke's s (Tove #9337 (1)). If R1 cannot
   finish its cap by 10:50 PDT it is cut at its last completed refit, scored there, and "decoy fell" counts as unmet
   (Kasper #9334); the post says so.
4. **Moving flag** (Tove #9337 (2)): the least-squares slope of catch over the last 4 refits, times 150 steps, has
   absolute value >= 10 pts. (Replaces "vs the 4 refits before", which barely exists at cap 300.)
5. **Wording** (Tove #9337 (2)): a lie that stays above 50% "held for the same N-step budget", shown with the full
   catch-vs-step curve; never "cannot be hidden".

## D9 — R2 restarted after a memory kill; full panel and every-position KL only at step 0 and the late refits (recorded 08:45 PDT, before any refit after step 0 was scored)

The first R2 (started 08:29:01) was killed by the 9000 MB watchdog at 08:42:17 inside its step-50 refit check: the
eval answer-KL built 800 x 151,936 fp32 logit arrays at once (Bruno #9347). Only its step-0 check existed (blind
100%, full 100%, output-only 100%: the clean model, as G0b'). R1 had started at 08:42 with the same bug and was killed
in its step-0 check, no data. Fix: KL in 100-row chunks, empty_cache after the capture; logs moved to `data/aborted/`.
R2 restarted 08:44:48 with seed 0 (a fresh run, not a resume). Time saver, same for both runs, no status number
touched: the FULL (unprojected) panel refit and the every-position KL slice are computed at step 0 and the late refits
150..300 only, not at 50/100. The blind status panel, the output-only probe, answer KL and behaviour are computed at
every refit. Projection: R2 ends ~09:40, R1 ~09:44-10:40, deadline guard 10:50.

## D10 — MPS pool cap; R1 time savers (recorded 09:01 PDT, before R1 started and before any R2 refit after step 50 was scored)

R2's second start was killed by the watchdog at 08:58:19 (9,149 MB) right after its step-50 refit (Bruno #9351);
the MPS caching allocator ratchets the footprint across train/check switches. Fix (numerics unchanged): heavy.sh sets
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.65 and PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5. R2 restarted 09:00:15, seed 0, fresh.
The killed run's step-50 numbers (blind 100%, output-only 100%, answer KL 0.058, first-token argmax unchanged on 64%
of lie prompts) are reported in the post's expert foldout as an aborted run, with no status role.
Projection: R2 ends ~09:57; R1 ~10:01-10:55, so the 10:50 guard would cut R1 at refit 250. R1 time savers (Kasper
#9343; R1 only; the status uses refits 150..300, which keep full scoring): (a) R1's step-0 scoring pass is skipped;
its step-0 catch is G0b' (`data/gate0/gate0b_spread.json`: the clean model, the same 640-prompt fit, line and eval);
(b) at R1's refits 50 and 100 the 800-prompt EVAL capture is skipped (the in-loop 640-prompt refit stays, so training
is identical). The figure shows R1's curve at step 0 (G0b') and from step 150 on, said in the caption.

## D11 — R1 deadline 10:50 -> 11:06 PDT so both runs reach the equal cap 300 (recorded 09:40 PDT, before R1 started; R2 at step 150)

Late refits cost ~565 s per 50 steps (Bruno #9360), so R2 ends ~10:09 and R1, with D10, would reach only refit 200 by
the 10:50 guard: its late mean would rest on 2 refits. Equal caps were frozen (D8 (3), Tove #9337 (1)), so R1's guard
moves to 11:06 PDT (Bruno's option A); its numbers land ~11:10, 50 min before the 12:00 hard end. The post text is
finished on R2 and every R1 number is a slot until then. Known when this was decided: R2's blind catch 100% at
steps 0-150 (not the late mean). If R1 is still cut, D8 (3) applies.
