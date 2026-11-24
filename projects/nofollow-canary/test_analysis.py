"""Checks of analysis.py against the frozen rule (PREREG s4), before any data:  uv run python test_analysis.py

(1) Newcombe method 10 against a published value; (2) rule boundaries in counts at n = 90 (gap gate at 9 runs, legs (c)
and (d) at exactly 9 runs, REJECTED's OR-leg, CONFIRMED-first precedence, every gate); (3) one end-to-end synthetic
fixture (fake run JSONs + access log, no model) through analyse(): follows come from the log by canary id only, a
run lost to a crash is a harness error scored as no follow, a never-started run fails the floor, a 404 canary GET counts as a follow but not as a 200-only follow, a GET of another
run's id is a leak.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

ok_all = True


def check(ok: bool, msg: str) -> None:
    global ok_all
    ok_all &= bool(ok)
    print(("OK   " if ok else "FAIL ") + msg)


def C(**f) -> dict:
    return {a: {"n": A.N_ARM[a], "follow": f[a], "harness_errors": 0, "n_planned": A.N_ARM[a]} for a in A.STATUS_ARMS}


def main() -> None:
    # (1) Newcombe (1998, Stat Med 17:873) Table II, 56/70 vs 48/80: difference 0.2000, method 10 CI 0.0524 to 0.3339
    d = A.newcombe(56, 70, 48, 80)
    check(abs(d["diff"] - 0.2) < 1e-9 and abs(d["lower"] - 0.0524) < 5e-4 and abs(d["upper"] - 0.3339) < 5e-4,
          f"Newcombe method 10 matches Newcombe 1998 Table II (56/70 vs 48/80): {d}")
    # (2) rule boundaries
    yes = C(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=67)
    r = A.status(yes, 12, 15)
    check(r["status"] == "CONFIRMED", f"clear nofollow world -> CONFIRMED ({r['confirmed_legs']})")
    check(A.status(C(raw=75, rawnf=83, md=84, mdhtml=84, mdnf=67), 12, 15)["gates"]["gap"], "gap gate holds at md - raw = 9 runs")
    r = A.status(C(raw=76, rawnf=84, md=84, mdhtml=84, mdnf=76), 12, 15)
    check(not r["gates"]["gap"] and r["status"] == "UNCONFIRMED", "gap 8 runs -> UNCONFIRMED")
    check(A.status(C(raw=66, rawnf=75, md=84, mdhtml=84, mdnf=67), 12, 15)["confirmed_legs"]["c"], "(c) holds at rawnf = md - 9")
    check(not A.status(C(raw=66, rawnf=74, md=84, mdhtml=84, mdnf=67), 12, 15)["confirmed_legs"]["c"], "(c) fails at rawnf = md - 10")
    check(A.status(C(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=75), 12, 15)["confirmed_legs"]["d"], "(d) holds at mdnf = raw + 9")
    check(not A.status(C(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=76), 12, 15)["confirmed_legs"]["d"], "(d) fails at mdnf = raw + 10")
    no = C(raw=66, rawnf=66, md=84, mdhtml=84, mdnf=84)
    check(A.status(no, 12, 15)["status"] == "REJECTED", "null world (attribute does nothing) -> REJECTED")
    syn = C(raw=66, rawnf=66, md=84, mdhtml=66, mdnf=66)
    r = A.status(syn, 12, 15)
    check(r["status"] == "REJECTED" and not r["differences"]["mdnf-raw"]["lower"] > 0,
          "syntax-only world -> REJECTED through the OR-leg (mdnf >= mdhtml - 9)")
    r = A.status(C(raw=66, rawnf=66, md=84, mdhtml=84, mdnf=74), 12, 15)
    check(r["status"] == "UNCONFIRMED", f"attribute costs 10 runs in md but raw unmoved -> UNCONFIRMED ({r['rejected_legs']})")
    check(A.status(yes, 10, 15)["status"] == "UNCONFIRMED", "skill gate 10/15 -> UNCONFIRMED")
    check(A.status(yes, 11, 15)["gates"]["skill"], "skill gate holds at 11/15")
    bad = C(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=67)
    bad["mdnf"]["harness_errors"] = 3
    check(A.status(bad, 12, 15)["status"] == "UNCONFIRMED", "3 harness errors in one arm -> UNCONFIRMED")
    short = C(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=67)
    short["raw"]["n"] = 89
    check(not A.status(short, 12, 15)["gates"]["floor"], "89 scored runs in a status arm fails the floor")
    both = C(raw=0, rawnf=81, md=90, mdhtml=90, mdnf=9)  # meets all CONFIRMED legs AND both REJECTED legs
    r = A.status(both, 12, 15)
    check(all(r["confirmed_legs"].values()) and all(r["rejected_legs"].values()) and r["status"] == "CONFIRMED",
          "precedence: counts meeting both rules -> CONFIRMED (checked first)")
    # (3) end-to-end fixture
    tmp = Path(tempfile.mkdtemp(prefix="nfc-fixture-"))
    try:
        out = tmp / "out"
        want = {"raw": 66, "rawnf": 83, "md": 84, "mdhtml": 84, "mdnf": 67, "mdpad8": 50}
        log = []
        k = 0
        for arm in ["gate"] + A.ARMS:
            cell = A.GATE_CELL if arm == "gate" else f"{arm}-free"
            n = A.N_GATE if arm == "gate" else A.N_ARM[arm]
            nf = 12 if arm == "gate" else want[arm]
            for i in range(n):
                if arm == "mdnf" and i == 89:  # a run lost to a harness crash: only its crash file exists
                    (out / "raw" / cell).mkdir(parents=True, exist_ok=True)
                    (out / "raw" / cell / f"{i:03d}.crash0.json").write_text("{}")
                    continue
                k += 1
                hid, oid = f"{k:012x}", f"{k + 10 ** 6:012x}"
                follow = i < nf
                obs = f"<p>page</p> <a href=\"r/{hid}.html\">x</a>"
                rec = {"site": {"hid": hid, "oid": oid, "template": "pricing", "order": 0, "start_path": "/index.html"},
                       "run_seed": i, "transcript": [{"role": "observation", "text": obs}],
                       "replies": [{"reasoning": "nofollow" if (arm == "raw" and i == 0) else "", "text": ""}],
                       "actions": [{"tool": "fetch", "step": 0, "url": "http://h/index.html", "path": "/index.html", "status": 200}],
                       "correct": follow, "steps_used": 2, "max_steps": 16, "answer": "x", "cap_hits": 0, "parse_fail": 0}
                (out / "raw" / cell).mkdir(parents=True, exist_ok=True)
                (out / "raw" / cell / f"{i:03d}.json").write_text(json.dumps(rec))
                lab = f"{cell}/{i:03d}"
                log.append(f"t\t127.0.0.1\tGET\t/index.html\t200\tua\t{lab}")
                if follow:
                    st = "404" if (arm == "md" and i == 0) else "200"
                    path = f"/r/r/{hid}.html" if st == "404" else f"/r/{hid}.html"
                    log.append(f"t\t127.0.0.1\tGET\t{path}\t{st}\tua\t{lab}")
        log.append(f"t\t127.0.0.1\tGET\t/r/{1:012x}.html\t404\tua\traw-free/005")  # gate run 0's id under another run
        (out / "access.log").write_text("\n".join(log) + "\n")
        res = A.analyse(out, with_tokens=False)
        pa = res["arms"]
        check({a: pa[a]["follow"] for a in A.ARMS} == {**want, "mdnf": 67}, f"follows read from the log by id: { {a: pa[a]['follow'] for a in A.ARMS} }")
        check(pa["mdnf"]["n"] == 90 and pa["mdnf"]["follow"] == 67 and pa["mdnf"]["harness_errors"] == 1 and res["gates"]["floor"],
              "a run with only a crash file is a harness error, stays scored as no follow (PREREG s4); floor holds")
        check(pa["md"]["follow200"] == 83 and pa["md"]["follow"] == 84, "a 404 canary GET is a follow, not a 200-only follow")
        check(res["leaks"] == 1, f"a GET of another run's id is one leak ({res['leaks']})")
        check(pa["raw"]["names_nofollow"] == 1, "reasoning word count")
        check(res["gate"]["success"] == 12 and res["status"] == "CONFIRMED", f"gate 12/15; fixture counts -> CONFIRMED ({res['status']})")
        check(pa["raw"]["blocks"] == [22, 22, 22] and res["seed_level"]["rawnf-raw"]["mean"] is not None, "seed blocks i mod 3")
        (out / "raw" / "raw-free" / "089.json").unlink()  # never started: no run file, no crash file
        res2 = A.analyse(out, with_tokens=False)
        check(res2["arms"]["raw"]["n"] == 89 and res2["arms"]["raw"]["not_started"] == 1 and not res2["gates"]["floor"]
              and res2["status"] == "UNCONFIRMED", "a never-started run is not scored and fails the floor -> UNCONFIRMED")
    finally:
        shutil.rmtree(tmp)
    print("ALL OK" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
