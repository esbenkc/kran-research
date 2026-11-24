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

## Still to do (continuation room, after laptop job tswdjfbrh2)

Commands, from `projects/nofollow-canary/` after `git pull --rebase --autostash`:

    python3 skeptics/skeptic/vera_check.py web/out/qwen3-8b > skeptics/skeptic/out_check.txt
    python3 skeptics/skeptic/vera_null.py web/out/qwen3-8b 20000 > skeptics/skeptic/out_null.txt

Then diff every count and Newcombe bound in `out_check.txt` against `results/analysis.json` (catch, differences,
seed_level, gates). Check that the gate runs (cell gatemd-free) are excluded from the arms. Then write `signoff.md`
(role / model / verdict lines first). That file does not exist yet, because no run of this room has finished.
