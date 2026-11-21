# Continuation — what is queued, what still has to be queued

State at the close of room oracle-1121-audit-penalty-8b (2026-11-20, PST). Code: this folder at the branch head.
Rule: `PREREG.md` (frozen e5de283a). Max 2 open labq jobs per room; every box writes only run- or box-unique files, so
two boxes' labq commits never conflict.

## Queued (laptop, Qwen/Qwen3-8B)

| job | runs | est | note |
|---|---|---|---|
| w3hckqmt2p (J1) | ao_s0 ao_s1 ao_s2 l0_s0, GUARD=ao | 3.5-4.5 h | first attempt lost at 18:21 PST (laptop slept 20 min into ao_s0); re-queued by labq |
| 1ew5iq4iqf (J1b) | l0_s1 | ~0.7 h | the 2nd L0 seed in its own short job (status-gating control) |

## Still to submit (in this order, when a slot frees; run from this folder on the room's branch)

```sh
# J2: original penalty, weight 100 (status cells; fake-no guard in the same box)
~/bin/labq submit --room <room> --title "Audit-penalty 8B, J2: original penalty weight 100 seeds 0-2" \
  --mem-gb 24 --est-hours 4.5 --disk-gb 4 --model Qwen/Qwen3-8B --output projects/probe-switch-audit-penalty-8b/data \
  --cmd "uv sync --frozen --no-install-project && RUNS='p100_s0 p100_s1 p100_s2' GUARD=p100 HOURS=6 caffeinate -i ./run.sh"
# J3: original penalty, weight 10 (no guard, PREREG) + the warm adapters' audits (reported rows) at the end
~/bin/labq submit --room <room> --title "Audit-penalty 8B, J3: original penalty weight 10 seeds 0-2 + warm audits" \
  --mem-gb 24 --est-hours 4.5 --disk-gb 4 --model Qwen/Qwen3-8B --output projects/probe-switch-audit-penalty-8b/data \
  --cmd "uv sync --frozen --no-install-project && RUNS='p10_s0 p10_s1 p10_s2' WARM='0 1 2' HOURS=6 caffeinate -i ./run.sh"
# J4: dev+test rows of the cache for the skeptic's rebuild (~250 MB gz; its own job so a failed push loses nothing else)
~/bin/labq submit --room <room> --title "Audit-penalty 8B, cache export (dev+test rows) for the skeptic rebuild" \
  --mem-gb 20 --est-hours 0.4 --disk-gb 4 --model Qwen/Qwen3-8B --output projects/probe-switch-audit-penalty-8b/data \
  --cmd "uv sync --frozen --no-install-project && EXPORT=1 HOURS=1 caffeinate -i ./run.sh"
```

## After the jobs land

`git pull --rebase --autostash`, then `AP_MODEL=Qwen/Qwen3-8B .venv/bin/python status.py` (mini, light: reads JSONs only)
-> `data/status.json`, `data/figure_rows.json`, `proof-matrix.md`. Box logs: `data/laptop/<job>/chain.log`.
A box that ended at its deadline leaves `data/ckpt/<name>.pt` and `run_state.txt` 'MORE ...': resubmit the same
command; it resumes. A box that logged `WARM GATE FAIL` means its cache does not carry the trigger: stop and compare
`data/laptop/<job>/cache_sha.txt` with `inputs/cache_sha_headclamp8b.txt`.

## Done on the mini (reported only)

`data1.7b/`: the 1.7B AO side row (parent warm adapters, parent cache): s0 strict 1.000 R_blind3 0.363, s1 1.000 /
0.467, s2 faded; warm R_blind3 0.902 / 0.966 / 0.990 (= the parent's published 0.90 / 0.97 / 0.99).
