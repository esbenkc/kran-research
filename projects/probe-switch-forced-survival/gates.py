"""Frozen gates of PREREG.md (probe-switch-forced-survival): tie test, seed gates, spare choice.

    python gates.py spare [--out data]   prints '<warm seed> <failed gate> <run>' for the spare, or nothing
    python gates.py show  [--out data]   prints the gates of every arm-F run that exists
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
FLOOR_GATE = 0.90      # FLOOR MET: final adapter, fp32 strict on all 512 train prompts
LIVE_GATE = 0.75       # LIVE: test S(0), fp32
G1_TOL = 0.05          # G1: cached vs whole-graph uncut argmax switch
LIVE_LEFT = 0.01       # a greedy step is LIVE when >= 1% of D is left before it
TIE_REL = 1e-3         # TIE: (best_j - ru_j) < 1e-3 x (best_j - best_{j-1})


def tie_flags(steps, D):
    """steps = [[best_j, runner-up_j], ...] of one joint greedy pick; D = the score of the whole switch (D^T_mean).
    Returns per step: left share before the step, live, tie (PREREG 'Tie test')."""
    out, prev = [], 0.0
    for j, (b, r) in enumerate(steps):
        left = (D - prev) / D if D > 0 else 0.0
        live = j == 0 or left >= LIVE_LEFT
        tie = r is not None and (b - r) < TIE_REL * (b - prev)
        out.append({"step": j + 1, "left": round(left, 6), "live": bool(live), "tie": bool(tie)})
        prev = b
    return out


def monitor_healthy(run: dict) -> bool:
    first = run["picks"][0]
    return not any(f["live"] and f["tie"] for f in tie_flags(first["greedy_best_runnerup"], first["D_mon"]))


def gates(name: str, out: Path) -> dict:
    """the five gates of one arm-F run; None = input missing (not finished)"""
    rp, wp, kp = out / "runs" / f"{name}.json", out / "whole" / f"{name}.json", out / "kaudit" / f"{name}.json"
    if not rp.exists():
        return {"run": name, "finished": False}
    r = json.loads(rp.read_text())
    g = {"run": name, "finished": True, "warm": r["warm"], "seed": r["args"]["seed"],
         "floor_met": r["train_final"]["switch_strict32"] >= FLOOR_GATE,
         "forced": r["floor_first_mu_pos_update"] is not None or r["mu_final"] > 0,
         "live": r["test_cached"]["switch_strict32"] >= LIVE_GATE,
         "monitor_healthy": monitor_healthy(r),
         "train_final_strict32": r["train_final"]["switch_strict32"], "S0_cached": r["test_cached"]["switch_strict32"],
         "share_mu_pos": r["floor_share_mu_pos"]}
    if wp.exists():
        w = json.loads(wp.read_text())
        g["g1_gap"] = abs(r["test_cached"]["switch_correct"] - w["switch_correct"])
        g["g1"] = g["g1_gap"] <= G1_TOL
    else:
        g["g1"] = None
    g["audited"] = kp.exists()
    names = ("floor_met", "forced", "live", "monitor_healthy", "g1")
    g["failed"] = [k for k in names if g[k] is False]
    g["counted"] = all(g[k] is True for k in names) and g["audited"]
    return g


def warm_seed(g: dict) -> int:
    return int(Path(g["warm"]).stem.split("_s")[1].split("_")[0])


def spare(out: Path):
    """PREREG: the spare replaces the FIRST seed in seed order that fails a gate; training seed 3; warm = that seed's
    own warm start, except MONITOR HEALTH -> the next warm start s0 -> s1 -> s2 -> s0."""
    for s in (0, 1, 2):
        g = gates(f"k6_s{s}", out)
        if g.get("finished") and g["failed"]:
            w = warm_seed(g)
            if "monitor_healthy" in g["failed"]:
                w = (w + 1) % 3
            return w, g["failed"][0], g["run"]
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spare", "show"])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = HERE / a.out
    if a.cmd == "spare":
        sp = spare(out)
        if sp:
            print(*sp)
    else:
        for n in ("k6_s0", "k6_s1", "k6_s2", "k6_s3"):
            print(json.dumps(gates(n, out)))
