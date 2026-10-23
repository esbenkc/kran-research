"""Print every number the post quotes, straight from data/results/results.json and theory.json, plus the
markdown table for the expert foldout. Run from projects/norm-rescale-free: python3 figure/post_numbers.py"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "data" / "results"
res = json.loads((RES / "results.json").read_text())
th = json.loads((RES / "theory.json").read_text())
runs = res["runs"]


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def last(r):
    return max(r["checkpoints"], key=lambda c: c["step"])


rows = []
for name in sorted(runs, key=lambda n: (runs[n]["lam"] != 1e4, runs[n]["arm"] != "pre", runs[n]["lam"], runs[n]["seed"])):
    r = runs[name]
    c = last(r)
    rows.append((name, r, c))
    print(f"{name}: step {c['step']} R {c['R']:.4f} drop {pct(c['gap_drop_frac'])} R_common {c['R_common']:.4f} "
          f"catch28 {pct(c['catch_reset'])} catch29 {pct(c['catch_reset_tok29'])} restored28 {pct(c['catch_reset_restored'])} "
          f"clean-line28 {pct(c['catch_clean'])} (FA {pct(c['false_alarm_clean'])}) KL {c['eval_kl']:.4f} "
          f"KLcountry {c['eval_kl_country']:.4f} E {pct(c['radial_energy_frac'])} norm f/t {c['norm_ratio_false']:.3f}/"
          f"{c['norm_ratio_true']:.3f} Rmax {max(x['R'] for x in r['checkpoints'] if x['R'] is not None):.4f}")

main = [x for x in rows if x[1]["lam"] == 1e4]
for arm in ("pre", "post"):
    a = [x for x in main if x[1]["arm"] == arm]
    if not a:
        continue
    f = lambda k: [x[2][k] for x in a]
    rng = lambda v, d=1: f"{pct(min(v), d)} to {pct(max(v), d)}"
    print(f"\nARM {arm} (n={len(a)}): R {rng(f('R'))}; drop {rng(f('gap_drop_frac'))}; catch28 {rng(f('catch_reset'))}; "
          f"catch29 {rng(f('catch_reset_tok29'))}; KL {min(f('eval_kl')):.4f} to {max(f('eval_kl')):.4f}; "
          f"shrink false {rng([1 - v for v in f('norm_ratio_false')])}")
print(f"\ntheory: R0 {th['R0']:.4f}, ideal {th['ideal_R']:.4f}, hand {th['analytic_catch_before']:.3f} -> "
      f"{th['analytic_catch_after']:.3f}, KL max {th['analytic_kl_max']:.1e}")

print("\n| Fine-tune | Run | Restore share | Gap drop, probe before | Caught, probe before | Caught, probe after | Output change |")
print("|---|---|---|---|---|---|---|")
lab = {"pre": "against the probe before", "post": "against the probe after"}
for name, r, c in rows:
    m, e = f"{r['lam']:.0e}".split("e")
    w = "" if r["lam"] == 1e4 else (", no output penalty" if r["lam"] == 0 else f", penalty weight {m} × 10" + str(int(e)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")))
    R = "n/a" if r["arm"] == "post" else pct(c["R"])
    print(f"| {lab[r['arm']]}{w} | {r['seed'] + 1} | {R} | {pct(c['gap_drop_frac'], 0)} | {pct(c['catch_reset'])} | "
          f"{pct(c['catch_reset_tok29'])} | {c['eval_kl']:.4f} |")
