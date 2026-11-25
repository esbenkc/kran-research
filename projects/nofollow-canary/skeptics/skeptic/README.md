# Skeptic (Vera) — nofollow-canary

Written by Vera (skeptic, Claude Opus via the subscription), room `defense-1124-length-or-markup`. Own code only. The
one build file used is the harness, as a black box for the view check, and as the format source for the fixture.

## Done before data (this room)

| file | what it does | result |
|---|---|---|
| `vera_power.py`, `vera_power2.py` | the scout's v1 point rule ('both within 10'), n=60 | P(YES): length 0.81, markup 0.01, half 0.34; law-miss clause 0.73 under every truth; CI-equivalence 0.000 |
| `vera_power3.py` … `vera_power6.py` | v1 / v1.1 rule forms (body leg, gap share) | body leg: P(CONF \| token-linear length) 0.15-0.20 |
| `vera_power7.py`, `vera_power8.py` | v2 nofollow rule, the mdhtml leg | old md leg: false YES 0.836 under 'syntax in md'; mdhtml leg 0.024 |
| `vera_power_final.py` → `out_power_final.txt` | the FROZEN rule's CONFIRMED form | true 0.833, null 0.000, no gap 0.000 |
| `vera_power_rej2.py` → `out_power_rej2.txt` | the REJECTED OR-leg (frozen, #29599) | syntax-only REJ 0.869 (draft form 0.023) |
| `vera_views.py` → `out_views.txt` | black-box diff of the 6 views on 80 canary pages | 0 FAIL; mdpad8/raw tokens 8.00-8.06 |
| `vera_null.py` | label-shuffle null of the whole frozen rule (attribute swap within run index; full shuffle) | on the 'true' fixture: P(CONFIRMED) 0.0005 under the attribute null |
| `vera_fixture.py`, `vera_check.py` → `out_fixture.txt` | 7 synthetic outdirs with planted counts at the frozen edges; my status vs `analysis.py` | 7/7 statuses agree; 42/42 counts match the plant |
| `novelty.md` | literature search for a nofollow ablation on LLM agents | none found (2 searches) |

The prereg stamp is checked: sha256 of `c6173ca68:PREREG.md` = `0d02e594…3fff`, which matches the header.

## Done after data (rooms defense-1124-length-or-markup continuation and defense-1125-nofollow-double)

Run from `projects/nofollow-canary/`. The verdict and all numbers are in `signoff.md`.

| file | what it does | result |
|---|---|---|
| `vera_rebuild.py web/out/qwen3-8b --null 2000` → `out_rebuild.txt` | fresh rebuild (2026-11-25 room) of the 5 status arms from `access.log`, Newcombe intervals, my copy of the frozen rule, arm-label shuffle within site index | 72/74/82/74/70 of 90, UNCONFIRMED; null: CONFIRMED 0/2000 |
| `vera_check.py` → `out_check.txt` | first landed rebuild + seed-level t-intervals | same counts and status |
| `vera_diff.py` → `out_diff.txt` | every field of `results/analysis.json` vs my rebuild | 127 fields, 0 mismatches |
| `vera_gates.py` → `out_gates.txt` | dedupe, id leaks, gates, harness-flag agreement | 0 duplicates, 0 leaks, gates hold |
| `vera_manip.py` → `out_manip.txt` | the canary markup the model actually saw | as designed in 90/90 runs per arm |
| `vera_landed_views.py` → `out_landed_views.txt` | raw vs rawnf and mdnf vs mdhtml pages equal after removing rel | 90/90 equal |
| `vera_null.py` → `out_null.txt` | label-shuffle null of the frozen rule (20,000 draws) | CONFIRMED 0.0001, REJECTED 0.0031 |
| `vera_paired.py` → `out_paired.txt` | paired exact McNemar per site index (sensitivity) | nofollow moves p 0.79 and 0.50 |
| `vera_edges.py` → `out_edges.txt` | one-arm edges and bootstrap of the status | 2 fewer rawnf follows flip to REJECTED |
| `vera_secondary.py` → `out_secondary.txt` | length arm (mdpad8) and miss reasons | mdpad8 58/60 |
| `vera_power_rep.py` → `out_power_rep.txt` | would a fresh replication settle it tonight | UNCONFIRMED 72-89% at 36-90 runs per arm |
