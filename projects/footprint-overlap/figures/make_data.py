"""Main figure data: one dot per monitor build per seed, x = overlap score (fixed before any behaviour row),
y = fooled share (flattery drop under that monitor / drop under the training probe), with 95% intervals.
Reads data/summary.json + data/overlap.json; writes figures/footprint-overlap/data.json (compact, 4 decimals).

    python3 figures/make_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent.parent
A = "ref@L14.q_norm"
SEEN = ("ref@L7.q_norm", "ref@L21.q_norm")


def r4(v):
    return None if v is None else round(float(v), 4)


def label(b: str) -> str:
    if b == "emu":
        return "rounding setting (no probe)"
    if b == A:
        return "training probe (layer 14)"
    kind, rest = b.split("@")
    L, site = rest.split(".", 1)
    site = {"q_norm": "query", "k_norm": "key", "v_proj": "value", "o_proj": "attention output", "mlp": "MLP output",
            "resid": "layer output"}.get(site.split("+")[0], site)
    k = {"ref": "probe", "clone": "buffer-copy probe", "score": "score-only probe"}[kind]
    return f"{k}, layer {L[1:]} {site}"


def main() -> None:
    S = json.load(open(HERE / "data" / "summary.json"))
    ov = json.load(open(HERE / "data" / "overlap.json"))["builds"]
    status = set(S["status"]["builds"])
    pts = []
    for h in ("s0@8192", "s1@8192"):
        P = S["policies"].get(h)
        if not P:
            continue
        for b, r in P["rows"].items():
            if b.startswith("none") or b not in ov:
                continue
            o = ov[b]
            kind = ("status" if b in status else "A" if b == A else "seen" if b in SEEN else
                    "bitexact" if o["class"] == "bitexact" else "other")
            pts.append({"seed": int(h[1]), "b": b, "name": label(b), "k": kind, "cls": o["class"],
                        "x": r4(o["overlap"]), "y": r4(r["share"]), "lo": r4(r["share_ci95"][0]),
                        "hi": r4(r["share_ci95"][1])})
    cells = {h: {"r2": r4(c["r2_fixed"]), "p": r4(c["perm_p"]), "n": S["status"]["n_status"]}
             for h, c in S["status"]["cells"].items()}
    terms = json.load(open(HERE / "figures" / "terms.json")) if (HERE / "figures" / "terms.json").exists() else {}
    data = {"pts": pts, "cells": cells, "status": S["status"]["status"], "terms": terms}
    out = HERE / "figures" / "footprint-overlap"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out / "data.json", "w"), separators=(",", ":"))
    print("wrote", out / "data.json", len(pts), "points", cells)


if __name__ == "__main__":
    main()
