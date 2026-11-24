"""The frozen status rule of PREREG.md (section 4) and the secondary claims (section 5), from the saved raw data only.

    uv run python analysis.py [--out web/out/qwen3-8b] [--res results]

Inputs (written by web/run.py): <out>/access.log (tab separated: time, ip, method, path, status, user agent, run
label), <out>/raw/<cell>/<NNN>.json (one run each; <NNN>.crash<k>.json = a harness crash that was rerun) and
<out>/manifest.jsonl. Follow is read from the access log only (a GET whose path holds the run's canary id); the run
JSON gives the canary id, the transcript (reach, reasoning words, token offsets) and the answer.
Outputs: <res>/analysis.json (status, gates, legs, intervals, per-arm counts, secondary claims) and
<res>/analysis_runs.csv (one row per scored or gate run). No model, no judge, no API.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "web"))

STATUS_ARMS = ["raw", "rawnf", "md", "mdhtml", "mdnf"]
ARMS = STATUS_ARMS + ["mdpad8"]
N_ARM = {"raw": 90, "rawnf": 90, "md": 90, "mdhtml": 90, "mdnf": 90, "mdpad8": 60}
GATE_CELL, N_GATE, GATE_MIN = "gatemd-free", 15, 11
MAX_HARNESS_ERRORS = 2
TENTH = Fraction(1, 10)
Z = 1.959964
T_DF2 = 4.303
PARENT_PRED_RAW = Fraction(80, 83)  # anchor-text-law results/analysis.json heldout B-raw (PREREG s5 S3)
ID = re.compile(r"/r/([0-9a-f]{12})\.html")


# ---- intervals (PREREG s4) -----------------------------------------------------------------------------------------

def wilson(x: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def newcombe(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Newcombe (1998) method 10, hybrid score interval for p1 - p2 at 95%."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return {"diff": d, "lower": d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            "upper": d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)}


def seed_level(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Difference of block-b catch rates (block = run index mod 3), t-interval over the 3 blocks (df 2)."""
    diffs = []
    for b in range(3):
        a = [r["follow"] for r in rows_a if r["i"] % 3 == b]
        c = [r["follow"] for r in rows_b if r["i"] % 3 == b]
        if not a or not c:
            return {"blocks": [], "mean": None, "lower": None, "upper": None}
        diffs.append(sum(a) / len(a) - sum(c) / len(c))
    m = statistics.mean(diffs)
    half = T_DF2 * statistics.stdev(diffs) / math.sqrt(3)
    return {"blocks": diffs, "mean": m, "lower": m - half, "upper": m + half}


# ---- data ----------------------------------------------------------------------------------------------------------

def read_log(out: Path) -> list[dict]:
    rows = []
    p = out / "access.log"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            rows.append({"time": f[0], "method": f[2], "path": f[3], "status": f[4], "ua": f[5], "run": f[6]})
    return rows


def _ntok():
    import agent  # the harness's own token count (Qwen3 tokenizer on json.dumps(text))
    return agent.ntok


def load(out: Path, with_tokens: bool = True) -> dict:
    log = [r for r in read_log(out) if r["method"] == "GET"]
    by_id: dict[str, list[dict]] = {}
    for r in log:
        for m in ID.finditer(r["path"]):
            by_id.setdefault(m.group(1), []).append(r)
    ntok = _ntok() if with_tokens else None
    cells = {a: f"{a}-free" for a in ARMS}
    cells["gate"] = GATE_CELL
    owner: dict[str, str] = {}
    data: dict[str, list[dict]] = {}
    for arm, cell in cells.items():
        n = N_GATE if arm == "gate" else N_ARM[arm]
        rows = []
        for i in range(n):
            p = out / "raw" / cell / f"{i:03d}.json"
            crashes = len(list((out / "raw" / cell).glob(f"{i:03d}.crash*.json"))) if (out / "raw" / cell).exists() else 0
            if not p.exists():
                # PREREG s4: a run lost to a harness crash after its reruns (crash files, no run file) stays scored as
                # no follow; a run with neither file was never started (it fails the floor gate)
                rows.append({"arm": arm, "i": i, "exists": False, "crashes": crashes, "scored": crashes > 0,
                             "harness_error": int(crashes > 0), "follow": 0, "follow200": 0, "seed_block": i % 3})
                continue
            rec = json.loads(p.read_text())
            site = rec["site"]
            hid, oid = site["hid"], site["oid"]
            owner[hid] = owner[oid] = f"{cell}/{i:03d}"
            gets = by_id.get(hid, [])
            obs = [t["text"] for t in rec["transcript"] if t["role"] == "observation"]
            first = next((o for o in obs if hid in o), None)
            reasoning = " ".join((x.get("reasoning") or "") + " " + (x.get("text") or "") for x in rec["replies"])
            fetched = [a for a in rec["actions"] if a.get("tool") == "fetch"]
            menu = [a for a in fetched if a.get("status") == 200 and not a["path"].startswith("/r/")
                    and a["path"] != site["start_path"]]
            req_steps = [a["step"] for a in fetched if hid in a.get("url", "")]
            rows.append({
                "arm": arm, "i": i, "exists": True, "scored": True, "crashes": crashes, "seed_block": i % 3,
                "harness_error": int(not rec.get("replies")),  # no model reply at all (PREREG s4)
                "run_seed": rec["run_seed"], "template": site["template"], "order": site["order"],
                "follow": int(bool(gets)),
                "follow200": int(any(g["status"] == "200" for g in gets)),
                "offtask": int(bool(by_id.get(oid))),
                "reach": int(first is not None),
                "correct": int(bool(rec["correct"])),
                "css": int(any(g["path"].endswith("/assets/site.css") and g["run"] == f"{cell}/{i:03d}" for g in log)),
                "first_hid_step": min(req_steps) if req_steps else -1,
                "menu_fetch": int(bool(menu)),
                "names_nofollow": int("nofollow" in reasoning.lower()),
                "steps": rec["steps_used"], "cap_hits": rec["cap_hits"], "parse_fail": rec["parse_fail"],
                "step_cap_hit": int(rec["steps_used"] >= rec["max_steps"] and rec["answer"] is None),
                "truncated": rec.get("truncated_calls", 0),
                "pad_short": int(any((a.get("view_info") or {}).get("pad_short") for a in fetched)),
                "tokens_before_hid": (ntok(first[:first.index(hid)]) if (ntok and first is not None) else -1),
            })
        data[arm] = rows
    # leaks: a canary or off-task id requested under another run's label
    leaks = [r for i_, rs in by_id.items() for r in rs if i_ in owner and r["run"] not in ("-", owner[i_])]
    return {"rows": data, "leaks": len(leaks), "log_gets": len(log)}


# ---- the frozen rule (PREREG s4) -------------------------------------------------------------------------------------

def counts(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["scored"]]
    return {"n": len(scored), "follow": sum(r["follow"] for r in scored),
            "harness_errors": sum(r["harness_error"] for r in scored), "n_planned": len(rows)}


def catch(c: dict) -> Fraction:
    return Fraction(c["follow"], c["n"]) if c["n"] else Fraction(0)


def status(c: dict[str, dict], gate_success: int, n_gate: int) -> dict:
    """c[arm] = {'n', 'follow', 'harness_errors', 'n_planned'}; returns gates, legs and the status (PREREG s4)."""
    k = {a: catch(c[a]) for a in STATUS_ARMS}
    nc = lambda a, b: newcombe(c[a]["follow"], c[a]["n"], c[b]["follow"], c[b]["n"])  # noqa: E731
    gates = {
        "skill": gate_success >= GATE_MIN and n_gate == N_GATE,
        "harness": all(c[a]["harness_errors"] <= MAX_HARNESS_ERRORS for a in STATUS_ARMS),
        "floor": all(c[a]["n"] == N_ARM[a] for a in STATUS_ARMS),
        "gap": k["md"] - k["raw"] >= TENTH,
    }
    d = {"rawnf-raw": nc("rawnf", "raw"), "mdhtml-mdnf": nc("mdhtml", "mdnf"),
         "md-rawnf": nc("md", "rawnf"), "mdnf-raw": nc("mdnf", "raw")}
    conf = {"a": d["rawnf-raw"]["lower"] > 0, "b": d["mdhtml-mdnf"]["lower"] > 0,
            "c": k["rawnf"] >= k["md"] - TENTH, "d": k["mdnf"] <= k["raw"] + TENTH}
    rej = {"a": d["md-rawnf"]["lower"] > 0,
           "b": d["mdnf-raw"]["lower"] > 0 or k["mdnf"] >= k["mdhtml"] - TENTH}
    if not all(gates.values()):
        st = "UNCONFIRMED"
    elif all(conf.values()):
        st = "CONFIRMED"
    elif all(rej.values()):
        st = "REJECTED"
    else:
        st = "UNCONFIRMED"
    return {"status": st, "gates": gates, "confirmed_legs": conf, "rejected_legs": rej, "differences": d,
            "catch": {a: float(k[a]) for a in STATUS_ARMS}, "gap_runs": c["md"]["follow"] - c["raw"]["follow"]}


def law_pred(x_md: float, x_raw: float, x_pad: float, md: float, raw: float, log: bool) -> float | None:
    f = (lambda v: math.log(v)) if log else (lambda v: v)
    if min(x_md, x_raw, x_pad) <= 0 or f(x_raw) == f(x_md):
        return None
    return min(1.0, max(0.0, md + (raw - md) * (f(x_pad) - f(x_md)) / (f(x_raw) - f(x_md))))


def analyse(out: Path, with_tokens: bool = True) -> dict:
    L = load(out, with_tokens)
    rows = L["rows"]
    c = {a: counts(rows[a]) for a in ARMS}
    gate_rows = [r for r in rows["gate"] if r["scored"]]
    gate_success = sum(1 for r in gate_rows if r["follow"] and r.get("correct"))
    res = status(c, gate_success, len(gate_rows))
    sc = {a: [r for r in rows[a] if r["scored"]] for a in ARMS}
    ex = {a: [r for r in sc[a] if r["exists"]] for a in ARMS}  # runs with a transcript (descriptive counts below)
    res["seed_level"] = {name: seed_level(sc[a], sc[b]) for name, (a, b) in
                         {"rawnf-raw": ("rawnf", "raw"), "mdhtml-mdnf": ("mdhtml", "mdnf"), "md-rawnf": ("md", "rawnf"),
                          "mdnf-raw": ("mdnf", "raw"), "md-raw": ("md", "raw")}.items()}
    res["gate"] = {"runs": len(gate_rows), "success": gate_success, "min": GATE_MIN}
    per_arm = {}
    for a in ARMS:
        s, e = sc[a], ex[a]
        n = len(s)
        f = sum(r["follow"] for r in s)
        miss = [r for r in e if not r["follow"]]
        x = [r["tokens_before_hid"] for r in e if r["tokens_before_hid"] >= 0]
        per_arm[a] = {
            "n": n, "follow": f, "catch": f / n if n else None, "wilson": wilson(f, n),
            "harness_errors": c[a]["harness_errors"], "crash_files": sum(r["crashes"] for r in rows[a]),
            "not_started": sum(1 for r in rows[a] if not r["scored"]), "pad_short": sum(r.get("pad_short", 0) for r in e),
            "follow200": sum(r["follow200"] for r in s), "offtask": sum(r["offtask"] for r in e),
            "reach": sum(r["reach"] for r in e), "correct": sum(r["correct"] for r in e), "css": sum(r["css"] for r in e),
            "names_nofollow": sum(r["names_nofollow"] for r in e),
            "misses": len(miss), "misses_menu_fetch": sum(r["menu_fetch"] for r in miss),
            "step_cap_hits": sum(r["step_cap_hit"] for r in e), "think_cap_hits": sum(r["cap_hits"] for r in e),
            "parse_fail_steps": sum(r["parse_fail"] for r in e), "truncated_calls": sum(r["truncated"] for r in e),
            "median_tokens_before_hid": statistics.median(x) if x else None,
            "blocks": [sum(r["follow"] for r in s if r["i"] % 3 == b) for b in range(3)],
        }
    res["arms"] = per_arm
    # S1 syntax, S2 length dose, S3 parent law clause, S5 200-only (PREREG s5)
    nc = lambda a, b, k="follow": newcombe(per_arm[a][k], per_arm[a]["n"], per_arm[b][k], per_arm[b]["n"])  # noqa: E731
    sec = {"S1_mdhtml-md": nc("mdhtml", "md")}
    pa = per_arm
    if all(pa[a]["n"] for a in ("md", "raw", "mdpad8")):
        xm, xr, xp = (pa[a]["median_tokens_before_hid"] or 0 for a in ("md", "raw", "mdpad8"))
        md_, raw_, pad_ = pa["md"]["catch"], pa["raw"]["catch"], pa["mdpad8"]["catch"]
        plog = law_pred(xm, xr, xp, md_, raw_, True)
        plin = law_pred(xm, xr, xp, md_, raw_, False)
        d = nc("md", "mdpad8")
        sec["S2_length"] = {
            "x_md": xm, "x_raw": xr, "x_mdpad8": xp, "catch_mdpad8": pad_, "md-mdpad8": d,
            "pred_log": plog, "pred_linear": plin,
            "log_law_fits": plog is not None and Fraction(pa["mdpad8"]["follow"], pa["mdpad8"]["n"]) <= Fraction(plog).limit_denominator(10 ** 9) + TENTH,
            "linear_law_fits": plin is not None and Fraction(pa["mdpad8"]["follow"], pa["mdpad8"]["n"]) <= Fraction(plin).limit_denominator(10 ** 9) + TENTH,
            "no_move_10pts": -0.1 < d["lower"] and d["upper"] < 0.1}
    if pa["raw"]["n"]:
        lo, hi = wilson(pa["raw"]["follow"], pa["raw"]["n"])
        err = pa["raw"]["catch"] - float(PARENT_PRED_RAW)
        sec["S3_law_clause"] = {"pred_raw": float(PARENT_PRED_RAW), "err": err, "lower": lo - float(PARENT_PRED_RAW),
                                "upper": hi - float(PARENT_PRED_RAW), "v1_wholly_below_minus10": hi - float(PARENT_PRED_RAW) < -0.10}
    sec["S5_200only"] = {a: pa[a]["follow200"] for a in ARMS}
    sec["S4_words"] = {a: {"names_nofollow": pa[a]["names_nofollow"], "n": pa[a]["n"],
                           "misses": pa[a]["misses"], "misses_menu_fetch": pa[a]["misses_menu_fetch"]} for a in ARMS}
    res["secondary"] = sec
    res["leaks"] = L["leaks"]
    res["log_gets"] = L["log_gets"]
    res["prereg_outcome"] = ("unsettled" if not all(res["gates"].values())
                             else "held" if res["status"] == "CONFIRMED" else "failed")
    res["_rows"] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web" / "out" / "qwen3-8b"))
    ap.add_argument("--res", default=str(ROOT / "results"))
    ap.add_argument("--no-tokens", action="store_true", help="skip the tokenizer (S2 x values become -1)")
    a = ap.parse_args()
    res = analyse(Path(a.out), not a.no_tokens)
    rows = res.pop("_rows")
    rd = Path(a.res)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    flat = [r for arm in ["gate"] + ARMS for r in rows[arm]]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with (rd / "analysis_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    pa = res["arms"]
    print(f"STATUS {res['status']}  gates {res['gates']}  gap {res['gap_runs']} runs  gate {res['gate']['success']}/{res['gate']['runs']}")
    for arm in ARMS:
        print(f"  {arm:7s} {pa[arm]['follow']}/{pa[arm]['n']}  errors {pa[arm]['harness_errors']}  blocks {pa[arm]['blocks']}")
    for k, v in res["differences"].items():
        print(f"  {k:12s} {v['diff']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]  seed-level {res['seed_level'][k]['mean'] if res['seed_level'][k]['mean'] is None else round(res['seed_level'][k]['mean'], 3)}")
    print(f"  CONFIRMED legs {res['confirmed_legs']}  REJECTED legs {res['rejected_legs']}  outcome {res['prereg_outcome']}")


if __name__ == "__main__":
    main()
