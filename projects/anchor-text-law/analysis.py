"""The frozen status rule of PREREG.md (sections 3-5) on the saved runs. Exact fractions; no model judge.

    uv run python analysis.py                 # reads web/out/A/<model>/ and web/out/B/<model>/, writes results/
    uv run python analysis.py --out-a DIR --out-b DIR --json results/x.json   (any two job folders)

Per run (one row in results/analysis_runs.csv):
  reach   the run's canary id hid is a substring of at least one observation message of the saved transcript
  follow  the job's access.log has a GET whose path contains hid (matched by id only, PREREG s3)
  off     the same for the off-task id oid (reported, never a follow); leak = a GET of a 12-hex id of another run
  css     a GET of /assets/site.css with this run's label (7th log column)
  cap     the run used all its steps without calling answer; harness_error = crash after reruns or no model reply
  pos     the canary's 0-based position in the first link list the run saw (bare/anchor), and the list length
Slot B's model is the one gate_decide kept (B/qwen3-8b-nothink, or B/qwen3-8b after the frozen swap).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "out"
FORMATS = ["raw", "md", "bare", "anchor"]
PLANNED = {"A": 60, "B": 42}
GATE_N, GATE_MIN, FLOOR, REACHED_FLOOR = 30, 21, 42, 20
TOL, GAP = F(1, 10), F(15, 100)
BOOT, BOOT_SEED = 2000, 20261123
T975_DF2 = 4.302652729911275  # scipy.stats.t.ppf(0.975, 2)
HEX = re.compile(r"r/([0-9a-f]{12})\.html")


# ---- per-run events ----------------------------------------------------------------------------------------------

def read_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        p = line.split("\t")
        if len(p) >= 6:
            rows.append({"method": p[2], "path": p[3], "status": p[4], "run": p[6] if len(p) > 6 else "-"})
    return rows


def list_position(rec: dict) -> tuple[int | None, int | None]:
    """Position of r/<hid>.html in the first 'Links:' list the run saw (bare/anchor views), and the list length."""
    hid = rec["site"]["hid"]
    for t in rec["transcript"]:
        if t["role"] == "observation" and "\n\nLinks:\n" in t["text"]:
            try:
                links = json.loads(t["text"].split("\n\nLinks:\n", 1)[1])
            except Exception:
                return None, None
            hrefs = [x["href"] if isinstance(x, dict) else x for x in links]
            for k, h in enumerate(hrefs):
                if hid in h:
                    return k, len(hrefs)
            return None, len(hrefs)
    return None, None


def job_rows(job: Path, slot: str) -> list[dict]:
    """One row per run index and cell of one job folder (final raw file, or a crash after reruns)."""
    log = read_log(job / "access.log")
    gets = [r for r in log if r["method"] == "GET"]
    rows = []
    for cell_dir in sorted((job / "raw").iterdir()) if (job / "raw").exists() else []:
        view, hiding = cell_dir.name.split("-", 1)
        idx = sorted({int(p.name[:3]) for p in cell_dir.glob("[0-9][0-9][0-9]*.json")})
        for i in idx:
            f = cell_dir / f"{i:03d}.json"
            crash = sorted(cell_dir.glob(f"{i:03d}.crash*.json"))
            if f.exists():
                rec = json.loads(f.read_text())
                err = len(rec.get("replies", [])) == 0
            else:
                rec = json.loads(crash[-1].read_text())
                err = True
            site = rec["site"]
            hid, oid = site["hid"], site["oid"]
            label = f"{cell_dir.name}/{i:03d}"
            obs = [t["text"] for t in rec.get("transcript", []) if t["role"] == "observation"]
            own = {hid, oid} | ({site["vid"]} if site.get("vid") else set())
            for cf in crash:  # a crashed attempt of this run built its own site (fresh ids): its GETs are not a leak
                own |= {json.loads(cf.read_text())["site"][k] for k in ("hid", "oid")}
            pos, n_links = list_position(rec) if f.exists() else (None, None)
            first = next((a["step"] for a in rec.get("actions", []) if a.get("tool") == "fetch" and hid in a.get("url", "")), None)
            rows.append({
                "slot": slot, "model": rec["model"], "cell": cell_dir.name, "format": view, "hiding": hiding, "i": i,
                "block": i % 3, "template": site["template"], "order": site["order"], "hid": hid,
                "reach": any(hid in o for o in obs),
                "follow": any(hid in g["path"] for g in gets),
                "follow_200": any(hid in g["path"] and g["status"] == "200" for g in gets),  # reported; not the event
                "off": any(oid in g["path"] for g in gets),
                "leak": any(m not in own for g in gets if g["run"] == label for m in HEX.findall(g["path"])),
                "css": any(g["run"] == label and g["path"].endswith("assets/site.css") for g in gets),
                "correct": bool(rec.get("correct")), "answered": rec.get("answer") is not None,
                "cap": (not err) and rec["steps_used"] >= rec["max_steps"] and rec.get("answer") is None,
                "harness_error": err, "steps": rec.get("steps_used"), "max_steps": rec.get("max_steps"),
                "first_hid_step": first, "pos": pos, "n_links": n_links, "crashes": len(crash),
            })
    return rows


# ---- frozen rule (pure functions on rows) ------------------------------------------------------------------------

def fr(cells: list[list[dict]]) -> F | None:
    reached = [r for c in cells for r in c if r["reach"]]
    return F(sum(r["follow"] for r in reached), len(reached)) if reached else None


def reach(c: list[dict]) -> F | None:
    return F(sum(r["reach"] for r in c), len(c)) if c else None


def catch(c: list[dict]) -> F | None:
    return F(sum(r["follow"] for r in c), len(c)) if c else None


def n_reached(c: list[dict]) -> int:
    return sum(r["reach"] for r in c)


def law(cell: dict) -> dict:
    """pred, catch and err of every held-out cell from cell[(slot, format)] -> rows (PREREG s4)."""
    fA = fr([cell["A", "raw"]])
    fB = fr([cell["B", "md"], cell["B", "anchor"]])
    out = {}
    for (m, f, other_fit) in (("A", "md", "B"), ("A", "anchor", "B"), ("B", "raw", "A"), ("B", "bare", "A")):
        follow = fA if m == "A" else fB
        r = reach(cell[other_fit, f])
        c = catch(cell[m, f])
        pred = None if follow is None or r is None else r * follow
        out[m, f] = {"pred": pred, "catch": c, "reach_from": r, "follow": follow,
                     "err": None if pred is None or c is None else c - pred}
    return out


def diffs(cell: dict) -> dict:
    out = {}
    for m in "AB":
        md, an, ba = fr([cell[m, "md"]]), fr([cell[m, "anchor"]]), fr([cell[m, "bare"]])
        out[m] = {"anchor_md": None if an is None or md is None else an - md,
                  "bare_md": None if ba is None or md is None else ba - md}
    return out


def pct(xs: list[F], q: F) -> F:
    """Percentile with linear interpolation (numpy's default), exact."""
    s = sorted(xs)
    k = q * (len(s) - 1)
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap(cell: dict, draws: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Runs resampled with replacement within every cell, all cells in one draw; err and fr differences recomputed."""
    rng = random.Random(seed)
    keys = sorted(cell)
    vals: dict = {}
    for _ in range(draws):
        d = {k: [cell[k][rng.randrange(len(cell[k]))] for _ in cell[k]] for k in keys}
        lw, df = law(d), diffs(d)
        for k, v in lw.items():
            vals.setdefault(("err",) + k, []).append(v["err"])
        for m, v in df.items():
            for name, x in v.items():
                vals.setdefault((name, m), []).append(x)
        for k in keys:  # reported only (figure whiskers): each cell's catch and fr in the same draw
            vals.setdefault(("catch",) + k, []).append(catch(d[k]))
            vals.setdefault(("fr",) + k, []).append(fr([d[k]]))
    ci = {}
    for k, xs in vals.items():
        good = [x for x in xs if x is not None]
        ci[k] = (pct(good, F(25, 1000)), pct(good, F(975, 1000)), len(xs) - len(good)) if good else (None, None, len(xs))
    return ci


def t_interval(xs: list[F]) -> tuple[float, float, float]:
    v = [float(x) for x in xs]
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    h = T975_DF2 * sd / math.sqrt(len(v))
    return m, m - h, m + h


def seed_level(cell: dict) -> dict:
    out: dict = {}
    per_b = []
    for b in range(3):
        d = {k: [r for r in v if r["block"] == b] for k, v in cell.items()}
        per_b.append((law(d), diffs(d)))
    for k in (("A", "md"), ("A", "anchor"), ("B", "raw"), ("B", "bare")):
        xs = [lw[k]["err"] for lw, _ in per_b]
        out["err", k] = {"blocks": xs, "t": t_interval(xs) if None not in xs else None}
    for m in "AB":
        for name in ("anchor_md", "bare_md"):
            xs = [df[m][name] for _, df in per_b]
            out[name, m] = {"blocks": xs, "t": t_interval(xs) if None not in xs else None}
    return out


def status(cell: dict, gate: dict, harness: dict, floor: dict, ci: dict) -> dict:
    """PREREG s4 FROZEN RULE, in its order. gate[m] = (successes, runs); harness[(m, f)] = (errors, runs)."""
    lw, df = law(cell), diffs(cell)
    nr = {k: n_reached(v) for k, v in cell.items()}
    ok = lambda *ks: all(nr[k] >= REACHED_FLOOR for k in ks)
    gates = {
        "skill": all(gate[m][0] >= GATE_MIN and gate[m][1] >= GATE_N for m in "AB"),
        "harness": all(F(e, n) <= F(2, 100) for e, n in harness.values()),
        "floor": all(floor[m] >= FLOOR for m in "AB"),
    }
    law_ok = ok(("A", "raw"), ("B", "md"), ("B", "anchor"))  # cells whose fr feeds a prediction
    g_dec = {m: ok((m, "bare"), (m, "md")) for m in "AB"}
    g_fired = {m: g_dec[m] and df[m]["bare_md"] <= -GAP for m in "AB"}
    a_dec = {m: ok((m, "anchor"), (m, "md")) for m in "AB"}
    gates["reached_floor"] = {"law": law_ok, "G": g_dec, "anchor": a_dec, "counts": {f"{m}-{f}": nr[m, f] for m, f in cell}}
    status_cells = [("A", "md"), ("A", "anchor"), ("B", "raw")]
    rej_a = law_ok and any(ci[("err",) + k][1] < -TOL or ci[("err",) + k][0] > TOL for k in status_cells)
    rej_b = any(g_fired[m] and a_dec[m] and ci["anchor_md", m][1] < -TOL for m in "AB")
    conf = (any(g_fired.values()) and law_ok and all(abs(lw[k]["err"]) <= TOL for k in status_cells)
            and all(a_dec[m] and abs(df[m]["anchor_md"]) <= TOL for m in "AB" if g_fired[m]))
    if not (gates["skill"] and gates["harness"] and gates["floor"]):
        st, why = "UNCONFIRMED", "a gate failed"
    elif rej_a or rej_b:
        st, why = "REJECTED", ("(a) a status-cell err interval is wholly beyond 10 points" if rej_a else "") + \
                  (" (b) anchor-md interval wholly below -10 points in a G model" if rej_b else "")
    elif conf:
        st, why = "CONFIRMED", "G holds, 3 status cells within 10 points, anchor within 10 of md in every G model"
    else:
        st, why = "UNCONFIRMED", "no G" if not any(g_fired.values()) else "a CONFIRMED clause failed or was undecidable"
    # prediction score (s5)
    c1 = law_ok and all(abs(lw[k]["err"]) <= TOL for k in status_cells)
    c2 = law_ok and lw["B", "bare"]["err"] <= -GAP
    c3 = all(a_dec[m] and abs(df[m]["anchor_md"]) <= TOL for m in "AB")
    gates_met = gates["skill"] and gates["harness"] and gates["floor"]
    outcome = "unsettled" if not gates_met else ("held" if c1 and c2 and c3 else "failed")
    return {"status": st, "why": why.strip(), "gates": gates, "G_fired": g_fired, "law_decidable": law_ok,
            "rejected_a": rej_a, "rejected_b": rej_b,
            "prediction": {"C1": c1, "C2": c2, "C3": c3, "outcome": outcome}}


# ---- report ------------------------------------------------------------------------------------------------------

def fx(x):
    return None if x is None else {"frac": f"{x.numerator}/{x.denominator}", "value": round(float(x), 6)}


def pick_b() -> Path:
    first = OUT / "B" / "qwen3-8b-nothink"
    dec = first / "gate_decision.json"
    if dec.exists() and json.loads(dec.read_text()).get("swap_to"):
        return OUT / "B" / json.loads(dec.read_text())["swap_to"]
    return first


def analyse(job_a: Path, job_b: Path) -> tuple[dict, list[dict]]:
    rows = job_rows(job_a, "A") + job_rows(job_b, "B")
    by = {}
    for r in rows:
        by.setdefault((r["slot"], r["cell"]), {})[r["i"]] = r
    floor, cell = {}, {}
    for m in "AB":
        n = 0
        while n < PLANNED[m] and all(n in by.get((m, f"{f}-free"), {}) for f in FORMATS):
            n += 1
        floor[m] = n - n % 3
        for f in FORMATS:
            cell[m, f] = [by[(m, f"{f}-free")][i] for i in range(floor[m])]
    gate = {m: (sum(r["follow"] and r["correct"] for i, r in by.get((m, "raw-none"), {}).items() if i < GATE_N),
                sum(1 for i in by.get((m, "raw-none"), {}) if i < GATE_N)) for m in "AB"}
    harness = {(m, f): (sum(r["harness_error"] for r in cell[m, f]), max(len(cell[m, f]), 1)) for m in "AB" for f in FORMATS}
    ci = bootstrap(cell) if all(cell.values()) else {}
    st = status(cell, gate, harness, floor, ci) if ci else {"status": "UNCONFIRMED", "why": "no scored runs yet"}
    lw, df = (law(cell), diffs(cell)) if all(cell.values()) else ({}, {})
    sl = seed_level(cell) if all(cell.values()) and min(floor.values()) >= 3 else {}
    cells_out = []
    for m in "AB":
        for f in FORMATS + ["gate", "bare-none", "anchor-none"]:
            name = {"gate": "raw-none"}.get(f, f if "-" in f else f + "-free")
            rs = cell.get((m, f)) if f in FORMATS else [r for r in by.get((m, name), {}).values()]
            if not rs:
                continue
            cells_out.append({
                "slot": m, "model": rs[0]["model"], "cell": name, "n": len(rs), "scored": f in FORMATS,
                "reached": n_reached(rs), "followed": sum(r["follow"] for r in rs),
                "followed_reached": sum(r["follow"] and r["reach"] for r in rs),
                "reach": fx(reach(rs)), "catch": fx(catch(rs)), "fr": fx(fr([rs])),
                "catch_ci": [fx(x) for x in ci[("catch", m, f)][:2]] if f in FORMATS and ci else None,
                "fr_ci": [fx(x) for x in ci[("fr", m, f)][:2]] if f in FORMATS and ci else None,
                "followed_200": sum(r["follow_200"] for r in rs),
                "correct": sum(r["correct"] for r in rs), "off_task": sum(r["off"] for r in rs),
                "css": sum(r["css"] for r in rs), "leak": sum(r["leak"] for r in rs),
                "cap_hits": sum(r["cap"] for r in rs), "harness_errors": sum(r["harness_error"] for r in rs),
                "per_block": {b: {"n": sum(r["block"] == b for r in rs), "reached": sum(r["reach"] for r in rs if r["block"] == b),
                                  "followed": sum(r["follow"] for r in rs if r["block"] == b),
                                  "followed_reached": sum(r["follow"] and r["reach"] for r in rs if r["block"] == b)} for b in range(3)},
                "canary_pos": sorted({(r["pos"], r["n_links"]) for r in rs if r["pos"] is not None}),
            })
    held = {}
    for k, v in lw.items():
        held[f"{k[0]}-{k[1]}"] = {**{kk: fx(vv) for kk, vv in v.items()},
                                  "err_ci": [fx(x) for x in ci[("err",) + k][:2]], "err_ci_undefined_draws": ci[("err",) + k][2],
                                  "err_seed": {"blocks": [fx(x) for x in sl["err", k]["blocks"]], "t": sl["err", k]["t"]} if sl else None}
    nulls = {}
    if lw:
        fA_blind = fr([cell["A", "raw"], cell["A", "bare"]])
        fB_blind = fr([cell["B", "md"], cell["B", "anchor"]])
        const = sum((catch(cell[k]) for k in (("A", "raw"), ("A", "bare"), ("B", "md"), ("B", "anchor"))), F(0)) / 4
        for k, v in lw.items():
            other = "B" if k[0] == "A" else "A"
            pb = reach(cell[other, k[1]]) * (fA_blind if k[0] == "A" else fB_blind)
            nulls[f"{k[0]}-{k[1]}"] = {"format_blind_pred": fx(pb), "format_blind_err": fx(v["catch"] - pb),
                                       "constant_pred": fx(const), "constant_err": fx(v["catch"] - const)}
    res = {
        "status": st, "floor": floor, "gate": {m: {"success": gate[m][0], "runs": gate[m][1]} for m in "AB"},
        "models": {m: (cell[m, "raw"][0]["model"] if cell.get((m, "raw")) else None) for m in "AB"},
        "heldout": held,
        "fr_diffs": {m: {name: {"value": fx(v), "ci": [fx(x) for x in ci[name, m][:2]] if ci else None,
                                "seed": {"blocks": [fx(x) for x in sl[name, m]["blocks"]], "t": sl[name, m]["t"]} if sl else None}
                         for name, v in df[m].items()} for m in df},
        "nulls": nulls, "cells": cells_out,
        "notes": {"followed_200": "follow (PREREG s3) is any GET whose path holds hid, whatever the status. followed_200 counts "
                  "only GETs that got the page (status 200), like the harness's own harness_followed field. They differ when "
                  "a run requests r/<hid>.html relative to a page inside r/ (path /r/r/<hid>.html, status 404): the request "
                  "names the canary, so it counts as a follow, but the agent never read the 2025 page."},
        "all_reach_ge_095": all(reach(v) >= F(95, 100) for v in cell.values()) if all(cell.values()) else None,
        "bootstrap": {"draws": BOOT, "seed": BOOT_SEED, "rng": "python random.Random, cells in sorted key order"},
        "jobs": {"A": str(job_a.relative_to(ROOT)), "B": str(job_b.relative_to(ROOT))},
    }
    return res, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-a", default=str(OUT / "A" / "gpt-oss-20b"))
    ap.add_argument("--out-b", default="")
    ap.add_argument("--json", default=str(ROOT / "results" / "analysis.json"))
    a = ap.parse_args()
    res, rows = analyse(Path(a.out_a).resolve(), Path(a.out_b).resolve() if a.out_b else pick_b())
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    with open(Path(a.json).with_name(Path(a.json).stem + "_runs.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    st = res["status"]
    print("STATUS", st["status"], "-", st["why"], "| floor", res["floor"], "| gate", res["gate"])
    for k, v in res["heldout"].items():
        print(f"  {k}: catch {v['catch'] and v['catch']['frac']} pred {v['pred'] and v['pred']['value']:.3f} "
              f"err {v['err'] and v['err']['value']:+.3f} ci {[x and round(x['value'], 3) for x in v['err_ci']]}")
    for m, d in res["fr_diffs"].items():
        print(f"  {m}: fr(anchor)-fr(md) {d['anchor_md']['value'] and d['anchor_md']['value']['value']}, "
              f"fr(bare)-fr(md) {d['bare_md']['value'] and d['bare_md']['value']['value']}")
    if "prediction" in st:
        print("  prediction", st["prediction"], "G", st["G_fired"])


if __name__ == "__main__":
    main()
