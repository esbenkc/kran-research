role: critic2
model: claude-opus-5-5
verdict: PASS

# critic2 sign-off, live-kl-off (room oracle-1118-live-kl-off-cont, 2026-11-18)

Claude fallback: the openai/gpt-5.5 critic2 hit its $2.46 cap ($2.51) before a verdict. This is a Claude check, not cross-family.

Answer to the question: YES (CONFIRMED). 3 settled seeds, 2 TAKE-OFF, 1 FLAT.

Rebuilt with my own code from raw data/runs/kl0_s{0,1,2}/reads/c_*.npy and data/pool/test/cols.npz (scripts and outputs in this folder; run twice, same numbers):
- s_P, all 32 reads of the 3 seeds: max |mine - reads.jsonl| = 0 (rebuild.out). Mean over the 2,788 rows without x1_orig; cheat_off over all 2,800.
- Frozen onset rule (both qualifying reads, start CI-high, paired CI-low > 0, prompt-clustered bootstrap 2,000): s0 no qualifying read (FLAT, last read 1,200, s_P 0.0101 [0.0080, 0.0123]); s1 qualifying 800, 900 -> onset (700, 800]; s2 qualifying 600, 700 -> onset (500, 600].
- Seed-level at the common read 700: 0.0048, 0.2185, 0.4537 -> mean 0.226, t-interval over 3 seeds [-0.332, 0.783]. Matches proof-matrix.md.
- Placebo ' Let' rise (test_let pool): s0 +0.0047, s1 +0.0014, s2 +0.0263; all < 0.10 -> guard passes (guards.out).
- Carrying gap (steps.jsonl, updates 1..last): 0.0994, 0.1074, 0.1048, all in [0.05, 0.15]. Probed ' We' share per 100-update block 0.074-0.132 (clause 2 holds per block too, trace.out).
- BROKEN: test cheat_off 0.40-0.65 at every read; max train probed cheat 0.906; no NaN; min 25-update mean of live probed groups 3.6.
- Trace s_R at each last read (own SNIS rebuild, D_R 0.10086): -0.0270 [-0.081, 0.025], -0.0299 [-0.099, 0.039], -0.0424 [-0.090, 0.005]; CI-high < 0.10 -> clause 3 holds.
- Use counts (sum n_probed_x): s2 2,014 by 600; s1 2,406 by 700, 2,786 by 800; s0 3,944 by 1,200 (and 1,970 by 600, where s2 had already taken off).
- Null test (null.py): prompt-clustered sign flip of the twin label: s2@600 obs 0.628 vs null sd 0.045; s1@800 0.847 vs 0.061; p < 1/2000 for both.

Prereg outcome held: clause 1 (2/3 take-off), clause 2, clause 3 all hold. The secondary predicted window (500, 700] caught seed 2, missed seed 1 (r = 800); the post says so.

Post (PR #222 head a7aba57, preview https://deploy-preview-222--kranai.netlify.app/live-kl-off): every number in TLDR, results, limits, outcome_note and the expert foldout matches the rebuild. Fisher split 2-of-3 vs 0-of-1 'half the time' = 2/4 = 0.50.

Open, not blocking: lab_cost_usd 3.54 vs room spend $3.56 at 14:25 PST; set it from 'room spend' at the merge.
