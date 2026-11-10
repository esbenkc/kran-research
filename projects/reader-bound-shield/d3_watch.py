"""DEVIATIONS D3 (Felix 1f0388a9, Tove #18674, Kasper #18675; frozen before the update-200 look result):
if the update-200 TEST look fails AND its TEST forced switch (c_notX - c_X, 800 CRN pairs) < +3 points, stop the gate
at 200: kill the running planted_s0 trainer, keep adapter_200.pt as the gate adapter, write D3_STOP. Else do nothing
(the run continues to 500 as frozen).   /usr/bin/python3 d3_watch.py   (polls gate.jsonl every 5 s)"""
import json, os, shutil, signal, subprocess, time
from pathlib import Path
R = Path(__file__).parent / "data" / "runs" / "planted_s0"
while True:
    rows = [json.loads(l) for l in open(R / "gate.jsonl")] if (R / "gate.jsonl").exists() else []
    g = [r for r in rows if r["step"] == 200]
    if g:
        g = g[0]
        break
    time.sleep(5)
fs = g["forced_switch"]
stop = (not g["pass"]) and fs is not None and fs < 0.03
print(f"D3: look 200 pass={g['pass']} y={g['test_gap']:.4f} forced={fs:.4f} -> stop={stop}", flush=True)
if stop:
    out = subprocess.run(["pgrep", "-f", "^[^ ]*python rl.py --arm planted .*--name planted_s0 "], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    for p in pids:
        os.kill(p, signal.SIGTERM)
    shutil.copy(R / "adapter_200.pt", R / "adapter.pt")
    (R / "D3_STOP").write_text(json.dumps({"look": g, "killed_pids": pids, "clock": time.strftime("%H:%M:%S"),
                                           "rule": "DEVIATIONS D3 1f0388a9"}, indent=1))
    print("D3: killed", pids, "adapter.pt = adapter_200.pt", flush=True)
