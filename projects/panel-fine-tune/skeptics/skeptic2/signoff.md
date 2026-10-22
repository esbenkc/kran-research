# Skeptic2 sign-off

Task: independent reproduction from saved raw data.

Verdict: SIGN-OFF: CONFIRMED for the pre-registered title metric.

What I rebuilt from raw data:

- `check.py` reads only `data/scores/*.npz`, `data/states/*/{kl,kl_pos0,lp,off}.npy`, `data/fits/{zmoments.json,panel.npz}`, and `data/ref/split.json`.
- I did not import Bruno's analysis code.
- Primary monitor: z-score each `tok10..tok29` score with clean-training true moments, take the max, set the alarm line to the 2nd-highest max score on the scored model's 200 held-out true statements.
- CI: 2,000 city-cluster bootstrap resamples over the 298 eval cities.
- Behaviour: recomputed eval KL from token KL arrays and right-country ranking from affirmative eval city pairs.
- Nulls: random directions at readout 11; shuffled-label refits for the best refit readout.

Primary numbers (`check.py:51-63`, `check.py:114-162`, output `out.json`):

- Max-panel catch, pooled over the three panel fine-tunes: 3.75%, CI [2.57%, 5.09%].
- Per seed: 2.18%, 6.21%, 2.85%.
- Eval false-alarm rates: 0.34%, 0.67%, 0.34%.
- Secondary 19-probe panel, without `tok29`: 3.58%, CI [2.40%, 4.87%].
- Clean-line diagnostic: 96.48% catch, but 66.67% false alarms on eval true statements. This is not the status metric.

Output checks (`check.py:165-204`, output `out.json`):

- Eval KL per seed: 0.00434, 0.00472, 0.00540 nats/token.
- Country-token KL mean per seed: 0.01019, 0.01090, 0.01127.
- Right-country rank: 298/298 for all three seeds.
- These pass the behaviour bars.

Important non-headline facts:

- The pre-registered prediction failed on clauses 2 and 4.
- Individual probes did not all fall below half-clean. Pooled own-line catches include tok10 42.2%, tok11 72.9%, tok12 48.2%, tok13 44.6%, tok14 32.9%, tok15 43.3%.
- The headline must say the max-panel was blinded, not that every probe was blinded.

Line lemma, independently derived and checked (`line_lemma.py`, output `line_lemma.json`):

- Derivation: for each row, `M_i = max_l z_{il}`. Since `M_i >= z_{il}` pointwise for every member `l`, the kth-highest line-setting value of `M` is at least the kth-highest line-setting value of each member. Therefore the panel line can be set by the loudest member, and a member with a lower own line can be silenced inside the max panel.
- Numeric check: `t_M >= max_l t_l` in clean and all three panel seeds.
- The line equals the loudest member in clean, s0 and s1; in s2 it is above the loudest member line by 0.117 z units.
- Tok11 own-line catch versus catch at the panel line: clean 98.0% / 94.6%; s0 71.8% / 0.0%; s1 71.0% / 0.0%; s2 76.0% / 0.0%.

Refit and null checks (`check.py:257-335`, output `out.json`):

- Refit recovery: best readout tok16 in all three seeds.
- Eval catch: 98.8%, 98.7%, 99.5%.
- OOF AUROC: 0.9984, 0.9986, 0.9988.
- Shuffled-label null max over 5 permutations: 2.7%, 4.7%, 3.4%.
- Random-direction null at tok11 on s0: mean 0.85%, 95th percentile 2.19%, max 4.70% over 100 random directions.

Norm route check (`out.json`):

- Readout-28 false median norm ratios are 0.969, 0.970, 0.968 across seeds.
- Readout-28 true median norm ratios are 0.961, 0.964, 0.961 across seeds.
- The fine-tune did not use the large free pre-norm rescale route from Iris's hand construction.

Warnings:

- `check.py` emitted NumPy/Accelerate RuntimeWarnings during dense matrix products. I checked `out.json` for non-finite values: 0 non-finite floats. The monitor, KL, rank, refit and null outputs are finite.

Definition check:

- CONFIRMED is valid only for the pre-registered max-panel monitor.
- It is not valid for the stronger statement that every individual probe is blind.
- It is not valid for a clean-model frozen alarm line.

Novelty check: see `novelty.md`.
