"""PREREG 'Budget' formula, evaluated ONCE at T0 (the clock when the planted run starts); writes data/gate/budget.json.

W = (21:00 PST - T0) - 72 min; N_p = min(100, floor_25(W / 3 / s)); N_r = min(200, floor_25((W - N_p s) / 2 / s));
if N_r < 75: s1 cut, N_r = min(200, floor_25((W - N_p s) / s)). s = s_update of the k64 smoke with --kl-prefix 0.1.

    python3 budget.py --t0           (at T0: writes budget.json, prints N_p)
    python3 budget.py --get N_r      (later lines: prints the frozen value)
"""
import argparse
import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data" / "gate" / "budget.json"
S_UPDATE = 16.91  # logs/time_k64_klp.log: 'TIME time_k64_klp 16.91 s/update after warm-up k=64 B=32'


def f25(x):
    return int(x // 25) * 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", action="store_true")
    ap.add_argument("--get")
    a = ap.parse_args()
    if a.get:
        print(json.loads(OUT.read_text())[a.get])
        return
    assert not OUT.exists(), "T0 already fixed"
    t0 = dt.datetime.now()
    end = t0.replace(hour=21, minute=0, second=0, microsecond=0)  # local clock = PST
    W = (end - t0).total_seconds() - 72 * 60
    n_p = min(100, f25(W / 3 / S_UPDATE))
    n_r = min(200, f25((W - n_p * S_UPDATE) / 2 / S_UPDATE))
    cut_s1 = n_r < 75
    if cut_s1:
        n_r = min(200, f25((W - n_p * S_UPDATE) / S_UPDATE))
    b = {"T0": t0.strftime("%Y-%m-%d %H:%M:%S PST"), "W_s": round(W, 1), "s_update": S_UPDATE, "N_p": n_p,
         "N_r": n_r, "cut_s1": cut_s1}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(b, indent=1))
    print(n_p)


if __name__ == "__main__":
    main()
