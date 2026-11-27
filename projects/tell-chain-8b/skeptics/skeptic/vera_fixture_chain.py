# Planted fixtures for vera_chain.py: synthetic chain JSONs (same keys as the smoke output) with known switch times.
import json, os, shutil, subprocess, sys
import numpy as np

SEEDS = (5, 6, 7, 8, 9)
CUE = {"main_C": "C", "ctrl_C": "C", "fresh_C": "C", "dose1_C": "C", "main_D": "D", "ctrl_D": "D", "fresh_D": "D"}


def mk(d, T, cens=(), unfinished=(), gb=0.004):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    rng = np.random.default_rng(0)
    base = list(0.35 + 0.01 * rng.standard_normal(64))
    meta = dict(p_base_off=base, gap_base={f"{s}{c}": gb for s in SEEDS for c in "ABCDE"},
                cache_sha256="70c4703aa2bab795eb5bc34b54be6c16ac3e62ec2ef361841c2598c351de67ba")
    json.dump(meta, open(f"{d}/run_meta.json", "w"))
    for kind, cue in CUE.items():
        for i, s in enumerate(SEEDS):
            t_sw = T[kind][i]
            c = (kind, s) in cens
            u = (kind, s) in unfinished
            cap = 584 if kind == "main_C" else 1536
            stop = cap if (c or kind == "main_C") else (400 if u else t_sw + 8)
            curve = []
            for t in range(0, stop + 1, 8):
                on = c or u or t < t_sw
                drop = 0.02 if on else 0.15  # D = drop - gb + gb... gap = drop + gb_true
                # a lone single-check spike below the switch must NOT count (needs two in a row)
                if not c and not u and t == t_sw - 40 and t_sw >= 48:
                    drop = 0.12
                curve.append(dict(t=t, p_off=base, **{"p_" + cue: [b - drop - gb for b in base]},
                                  p_E=[b - gb for b in base]))
            json.dump(dict(curve=curve, cache_sha256=meta["cache_sha256"],
                           T=None if (c or u or (kind == "main_C" and t_sw > 584)) else t_sw,
                           end_reason="wall" if u else ("cap" if c else "switch")),
                      open(f"{d}/{kind}_s{s}.json", "w"))


def ref(v):
    return {k: list(v) for k in CUE}


fresh = [296, 280, 304, 288, 296]
cases = {}
mult = ref(fresh); mult["main_C"] = [160, 152, 168, 160, 160]; mult["ctrl_C"] = [296, 288, 296, 280, 296]
mult["main_D"] = [120, 112, 120, 128, 120]; mult["ctrl_D"] = [296, 288, 296, 288, 296]
cases["F1_mult"] = (mult, (), (), "CONFIRMED", "held")
flat = ref(fresh); flat["main_C"] = [216, 216, 224, 208, 216]; flat["main_D"] = [216, 208, 224, 216, 216]
cases["F2_flat"] = (flat, (), (), "REJECTED", "failed")
mid = ref(fresh); mid["main_C"] = [184, 176, 184, 176, 184]; mid["main_D"] = [152, 160, 144, 168, 152]
cases["F3_mid"] = (mid, (), (), "UNCONFIRMED", None)
cases["F4_cens2_ctrlD"] = (mult, (("ctrl_D", 5), ("ctrl_D", 6)), (), "UNCONFIRMED", None)
port = ref([400, 392, 408, 400, 416]); port.update({k: mult[k] for k in ("main_C", "ctrl_C", "main_D", "ctrl_D")})
cases["F5_portfail"] = (port, (), (), "UNCONFIRMED", None)
cases["F6_3seeds_unfinished"] = (mult, (), (("main_D", 8), ("main_D", 9)), "CONFIRMED", None)
cases["F7_2seeds_unfinished"] = (mult, (), (("main_D", 7), ("main_D", 8), ("main_D", 9)), "UNCONFIRMED", None)

here = os.path.dirname(os.path.abspath(__file__))
ok = 0
for name, (T, cens, unf, want, want_pred) in cases.items():
    d = f"/tmp/vera_fix_chain/{name}"
    mk(d, T, cens, unf)
    out = subprocess.run([sys.executable, os.path.join(here, "vera_chain.py"), d], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("STATUS")][0]
    got = line.split()[1]; pred = line.split()[3]
    good = got == want and (want_pred is None or pred == want_pred) and "T mismatches vs stored: 0" in out
    ok += good
    print(f"{name:22s} want {want:11s} got {got:11s} pred {pred:9s} {'OK' if good else 'FAIL'}")
    if not good:
        print(out)
print(f"{ok}/{len(cases)} fixtures OK")
