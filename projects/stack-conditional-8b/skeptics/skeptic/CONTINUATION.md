# Skeptic sign-off steps for the continuation (after the laptop job lands)

From projects/stack-conditional-8b/skeptics/skeptic, after `git pull --rebase --autostash`:

1. `python3 vera_check.py ../../data/eval --step 8192` -> status, per-run dGAP/CI/drops/YES/NO, tail dGAP (recorder
   only), build-label shuffle null per run; writes rebuild.json (Bruno's proof-matrix generator adds one
   'rebuild (Vera)' row per run and flags |diff| > 1e-6).
2. `python3 vera_check.py ../../data/eval --step 4096` -> trajectory read (not in the status).
3. `python3 vera_onset.py ../../data/runs ../../data/t_pred.json ../../data/rows02_dev.npz` -> s from raw bf16 rows,
   rows 0-2 identical across prompts?, per-row squared-offset profile, T_naive, T_x33, T_obs per run, scored clause.
4. Check gates.json (G1 >= 190/200 both clauses, G2 dev P in 0.2-0.7), g1b.json, and that every run's
   data/runs/<run>.json has steps == 8192 and the dev curve reaches 8192.
5. Compare with data/status.json field by field; any status difference is a REJECT until explained.
6. Write signoff.md (role: skeptic / model / verdict), then
   `room review <room> --as skeptic "<build's result task>" --pass "SIGN-OFF: ..."` or `--fail "REJECT: ..."`.
