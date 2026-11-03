"""Fidelity gate: the auditor kernel's run() modes must reproduce the P2 screen bit-for-bit (logit sha per prompt).

Starts the session's kernel exactly as mcp_server.py does (sandbox-exec, same env), feeds it the 200 test prompts
through a temporary file in its home dir (deleted after), and compares run(mid, ..., mode, 'logits') with
data/screen/px_<build>.json for every model of the session and every mode the arm has.

    ~/.aud/venv/bin/python kran-research/projects/build-switch-audit/harness/gate_kernel.py --arm B --seed 99
writes data/sessions/<arm>_s<seed>/gate_kernel.json
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
PROJ = HERE.parent

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
kd = PROJ / "data" / "sessions" / f"{a.arm}_s{a.seed}"
key = json.load(open(kd / "key.json"))
sd = Path(key["session_dir"])
cd = Path(key["control_dir"])
cfg = json.load(open(cd / "session.json"))
tp = [json.loads(l)["text"] for l in open(PROJ / "src_cham" / "raw" / "prompts.jsonl")
      if json.loads(l)["split"] == "test"][: a.n]
tmpf = sd / "home" / "gate_prompts.json"
tmpf.write_text(json.dumps(tp))
env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(sd / "home"), "PYTHONNOUSERSITE": "1",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(sd / "tmp"),
       "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
if any(m != "eager" for m in cfg["modes"]):
    env.update({"TORCHINDUCTOR_CACHE_DIR": str(sd / "tmp" / "inductor"), "TORCHINDUCTOR_COMPILE_THREADS": "1"})
p = subprocess.Popen(["sandbox-exec", "-f", str(cd / "sandbox.sb"), cfg["python"], "-u", str(sd / "kernel.py"),
                      "--session-dir", str(sd), "--modes", ",".join(cfg["modes"])], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, text=True, cwd=str(sd / "home"), env=env)


def call(code):
    p.stdin.write(json.dumps({"code": code}) + "\n")
    p.stdin.flush()
    r = json.loads(p.stdout.readline())
    if r["error"]:
        raise RuntimeError(r["out"])
    return r["out"]


call("import json, hashlib; TP = json.load(open('gate_prompts.json'))")
build_of = {"eager": "eager", "compiled": "compiled", "compiled_probe": "compiled_probe"}
res = {}
for mode in cfg["modes"]:
    scr = json.load(open(PROJ / "data" / "screen" / f"px_{build_of[mode]}.json"))
    for mid, cid in key["ids"].items():
        out = call(f"L = run({mid!r}, TP, mode={mode!r}, output='logits')\n"
                   "print(json.dumps([hashlib.sha256(r.to(torch.bfloat16).contiguous().view(torch.int16).numpy()"
                   ".tobytes()).hexdigest()[:16] for r in L]))\n"
                   f"print(float(run({mid!r}, TP, mode={mode!r}, output='px').mean()))")
        lines = out.strip().splitlines()
        js = [l for l in lines if l.startswith('["')]
        if not js:
            raise RuntimeError(out[-3000:])
        shas, pxm = json.loads(js[-1]), float(lines[-1])
        ref = scr[cid]["sha"][: a.n]
        same = sum(x == y for x, y in zip(shas, ref))
        res[f"{mode}/{cid}"] = {"sha_equal": same, "n": len(ref), "px_kernel": pxm,
                                "px_screen": float(sum(scr[cid]["px"][: a.n]) / len(ref))}
        print(mode, cid, res[f"{mode}/{cid}"], flush=True)
call("o = run(MODELS[0], PROMPTS[:2], mode=RUN_MODES[-1], output='attn', layers=[26, 27])\nprint(len(o), o[0][27].shape)")
import re as _re
_fp = subprocess.run(["footprint", str(p.pid)], capture_output=True, text=True).stdout
res["kernel_footprint_after_all_modes"] = _re.search(r"Footprint:\s*([\d.]+\s*\w+)", _fp).group(1)
print("KERNEL FOOTPRINT", res["kernel_footprint_after_all_modes"])
p.stdin.close()
p.wait(timeout=60)
tmpf.unlink()
json.dump(res, open(kd / "gate_kernel.json", "w"), indent=1)
print("ALL EQUAL" if all(v["sha_equal"] == v["n"] for v in res.values() if isinstance(v, dict)) else "MISMATCH")
