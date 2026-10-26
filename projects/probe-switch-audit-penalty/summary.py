"""Collect every run into data/ (git) and write data/summary.json + proof-matrix.md rows + the status rule (PREREG).

Inputs (raw/, local): raw/step2/Qwen3-1.7B/runs/<name>.json (train curve + cached test rows),
raw/step2/Qwen3-1.7B/whole/<name>__emu0.json (whole compiled graph, per prompt), data/audit/<name>.json (audit.py).
Outputs (git): data/runs/<name>.json (copy without nothing dropped), data/whole/<name>.json, data/summary.json.

    $PY summary.py
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw" / "step2" / "Qwen3-1.7B"
D = HERE / "data"


def main() -> None:
    (D / "runs").mkdir(parents=True, exist_ok=True)
    (D / "whole").mkdir(parents=True, exist_ok=True)
    models = {}
    for f in sorted((RAW / "runs").glob("*.json")):
        name = f.stem
        if name.startswith("smoke"):
            continue
        shutil.copy(f, D / "runs" / f.name)
        r = json.loads(f.read_text())
        a = r["args"]
        m = {"name": name, "seed": a["seed"], "lambda": a["lam"], "headfree": a["headfree"], "updates": a["steps"] + a.get("start_step", 0),
             "warmup": a.get("warmup", 0), "p0": r.get("p0"), "coef": r.get("coef"),
             "cached_switch": r["test_cached"]["switch_correct"], "cached_push": r["test_cached"]["shift_mean"]}
        last = r["curve"][-1]["mon"]
        m["dev_monitor_last"] = {k: last[k] for k in ("switch_all32cut", "push_removed_all32", "blind_top3_single",
                                                     "push_removed_max_single", "kl_full", "shift_mean")}
        if r.get("grad_log"):
            g = r["grad_log"]
            m["grad_ratio_pen_over_ce_first20_median"] = sorted(x["gnorm_lam_pen"] / max(x["gnorm_ce"], 1e-12)
                                                                for x in g[:20])[len(g[:20]) // 2]
        w = RAW / "whole" / f"{name}__emu0.json"
        if w.exists():
            shutil.copy(w, D / "whole" / f"{name}.json")
            wj = json.loads(w.read_text())
            m |= {"switch_rate": wj["switch_correct"], "unprobed_B_rate": wj["share_unprobed_B"],
                  "probed_A_rate": wj["share_probed_A"], "push_mean_whole": wj["shift_mean"]}
            m["G1_cached_vs_whole_ok"] = abs(m["cached_switch"] - m["switch_rate"]) <= 0.05
        au = D / "audit" / f"{name}.json"
        if au.exists():
            aj = json.loads(au.read_text())
            h, mp = aj["headline"], aj.get("mlp_path", {})
            m |= {"R_blind3": h["title_cell_blind_top3_removed"], "R_informed3": h.get("informed_top3_removed"),
                  "R_moves3": h.get("moves_top3_removed"), "R_all32": h["all_32_removed"],
                  "switch_all32cut": h["switch_all32cut_D2"], "R_reverse3": h["reverse_top3_recovered"],
                  "null_p95": h["null_p95"], "push_mean_cached": h["push_mean"],
                  "identity_ok": h.get("identity_ok"), "mlp_share": mp.get("mlp_share_D1"),
                  "skip_leftover": mp.get("skip_leftover_D1"),
                  "clause3_ratio": mp.get("prereg_clause3_M_over_1_minus_Rblind3"),
                  "clean_kl_all_pos": aj.get("clean_kl_vs_ref", {}).get("clean_kl_all_pos_mean")}
        models[name] = m

    # Prediction (PREREG, scored at lambda 100) and status (D5: at ANY lambda in {1,10,100} with >= 2 finished
    # seeds, >= 2 seeds each with switch >= 0.75 AND R_blind3 < 0.30). The fake-no guard is applied by hand.
    def cell(m):
        return m["switch_rate"] >= 0.75 and m["R_blind3"] < 0.30
    per_lam = {}
    for lam in (1, 10, 100):
        ready = [m for m in models.values() if m["lambda"] == lam and not m["headfree"]
                 and "switch_rate" in m and "R_blind3" in m]
        per_lam[lam] = {"finished": sorted(m["name"] for m in ready),
                        "passing": sorted(m["name"] for m in ready if cell(m))}
    counted = {l: v for l, v in per_lam.items() if len(v["finished"]) >= 2}
    if any(len(v["passing"]) >= 2 for v in counted.values()):
        status = "CONFIRMED"
    elif counted:
        status = "REJECTED (subject to the fake-no guard and unfinished lambdas)"
    else:
        status = "UNSETTLED"
    warm = {m["seed"]: m for m in models.values() if m["name"].startswith("warm_")}
    clause4 = {m["name"]: (m["unprobed_B_rate"] - warm[m["seed"]]["unprobed_B_rate"]) * 100
               for m in models.values() if m["lambda"] == 100 and "unprobed_B_rate" in m
               and m["seed"] in warm and "unprobed_B_rate" in warm[m["seed"]]}
    out = {"status_rule_raw": status, "per_lambda": per_lam,
           "clause4_unprobedB_drop_points": clause4, "models": models}
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    for m in models.values():
        print(m["name"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                          if k in ("switch_rate", "unprobed_B_rate", "R_blind3", "R_informed3", "R_all32",
                                   "switch_all32cut", "R_reverse3", "mlp_share", "cached_switch", "cached_push")})
    write_matrix(models)


def write_matrix(models: dict) -> None:
    """proof-matrix.md: one row per cell; artifact paths relative to the project folder."""
    rows = []
    for m in sorted(models.values(), key=lambda m: (m["headfree"], -m["lambda"], m["seed"], m["name"])):
        n, s = m["name"], m["seed"]
        if n.startswith("warm_"):
            claim, variant, control = "baseline", "warm start (head-route switch, no extra training)", "start model"
        elif m["headfree"]:
            claim, variant, control = "head-free existence (A0)", f"all 32 heads swapped in training, {m['updates']} upd", "none"
        elif m["lambda"] == 0:
            claim, variant, control = "control", f"weight 0, {m['updates']} upd", "no penalty, same updates"
        else:
            claim = "headline" if m["lambda"] == 100 else f"weight {m['lambda']:g}"
            variant, control = f"weight {m['lambda']:g}, {m['updates']} upd", "none"
        if "switch_rate" in m:
            rows.append((claim, s, variant, control, f"switch rate (whole graph) {m['switch_rate']:.3f}; unprobed-B "
                         f"{m['unprobed_B_rate']:.3f}", f"data/whole/{n}.json"))
        if "R_blind3" in m:
            f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
            rows.append((claim, s, variant, control,
                         f"blind top-3 removes {f(m['R_blind3'])}; informed {f(m['R_informed3'])}; all 32 "
                         f"{f(m['R_all32'])}; switch with all 32 cut {f(m['switch_all32cut'])}",
                         f"data/audit/{n}.json"))
            rows.append((claim, s, variant, "random 3-head sets (null, 50 draws)", f"p95 removed {f(m['null_p95'])}",
                         f"data/audit/{n}.json"))
            if m.get("identity_ok") is not None:
                rows.append((claim, s, variant, "identity swap (batch-shape rounding)", f"ok={m['identity_ok']}",
                             f"data/audit/{n}.json"))
        rows.append((claim, s, variant, control, f"cached-tail switch {m['cached_switch']:.3f}, push "
                     f"{m['cached_push']:.2f} nats", f"data/runs/{n}.json"))
    lines = ["# Proof matrix — probe-switch-audit-penalty", "",
             "Generated by `summary.py` from data/. Skeptic rebuild cells go below the marker.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    lines += [f"| {c} | {s} | {v} | {ct} | {r} | `{a}` |" for c, s, v, ct, r, a in rows]
    old = (HERE / "proof-matrix.md").read_text() if (HERE / "proof-matrix.md").exists() else ""
    keep = old.split("<!-- skeptic rows -->", 1)[1] if "<!-- skeptic rows -->" in old else "\n"
    (HERE / "proof-matrix.md").write_text("\n".join(lines) + "\n\n<!-- skeptic rows -->" + keep)


if __name__ == "__main__":
    main()
