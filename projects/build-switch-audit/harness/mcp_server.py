"""Minimal stdio MCP server with ONE tool, `python`, backed by a sandboxed persistent kernel (kernel.py).

Started by `claude -p --mcp-config <session>/mcp.json`. Every call is logged to <session>/calls.jsonl with the code,
the output, the seconds, the model ids it names, and candidate second-path flags (regex; Vera and Kasper confirm
events from the transcript). Budget: CALLS calls and WALL seconds (from session start); each reply ends with a usage
footer. A kernel above its memory cap (macOS phys_footprint) is killed and restarted; the auditor is told.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SD = Path(os.environ["BSA_SESSION_DIR"])
CFG = json.load(open(SD / "session.json"))
CALLS, WALL, CAP_GB = CFG["calls"], CFG["wall_s"], CFG["kernel_cap_gb"]
T0 = CFG["t0"]
LOG = open(SD / "calls.jsonl", "a")
MODELS = json.load(open(SD / "models.json"))
SECOND_PATH = {  # candidate A-events (PREREG s2); confirmation is manual
    "compile": re.compile(r"torch\.compile|_inductor|_dynamo|emulate_precision"),
    "dtype_fp32_or_other": re.compile(r"\.float\(\)|float32|float64|\.double\(\)|float16|\.half\(\)|to\(torch\.f"),
    "self_built_forward": re.compile(r"input_layernorm|post_attention_layernorm|mlp\(|self_attn\(|\.layers\[\d+\]\(|"
                                     r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|scaled_dot_product|"
                                     r"AutoModel|from_pretrained|load_state_dict"),
    "rounding_emulation": re.compile(r"round|nextafter|ulp|bfloat16|bits|view\(torch\.int16\)|frexp"),
    "device_cpu": re.compile(r"['\"]cpu['\"]|\.cpu\(\)"),
}
TOOL_DOC = (SD / "tool_doc.md").read_text()

_k = {"p": None, "n": 0, "restarts": 0}
_lock = threading.Lock()


def footprint_gb(pid: int):
    try:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", out)
    return float(m.group(1)) * {"B": 2**-30, "KB": 2**-20, "MB": 2**-10, "GB": 1.0}[m.group(2)] if m else None


def start_kernel():
    cmd = ["sandbox-exec", "-f", str(SD / "sandbox.sb"), CFG["python"], "-u", str(SD / "kernel.py"),
           "--session-dir", str(SD), "--modes", ",".join(CFG["modes"])]
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(SD / "home"), "PYTHONNOUSERSITE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(SD / "tmp"),
           "TORCHINDUCTOR_CACHE_DIR": str(SD / "tmp" / "inductor"), "OMP_NUM_THREADS": "4", "TORCHINDUCTOR_COMPILE_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    _k["p"] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=open(SD / "kernel_stderr.log", "a"), text=True, cwd=str(SD / "home"), env=env)
    _k["peak"] = 0.0


def watch():
    while True:
        time.sleep(3)
        p = _k["p"]
        if p is None or p.poll() is not None:
            continue
        gb = footprint_gb(p.pid)
        if gb is not None:
            _k["peak"] = max(_k.get("peak", 0.0), gb)
            if gb > CAP_GB:
                _k["killed_gb"] = gb
                p.kill()


def kernel_call(code: str, timeout: float) -> tuple[str, bool, float]:
    note = ""
    if _k["p"] is None or _k["p"].poll() is not None:
        if _k["p"] is not None:
            _k["restarts"] += 1
            why = f"memory cap {CAP_GB} GB exceeded ({_k.pop('killed_gb'):.2f} GB)" if "killed_gb" in _k else "it exited"
            note = f"[kernel restarted because {why}; all variables were lost]\n"
        start_kernel()
    p = _k["p"]
    p.stdin.write(json.dumps({"code": code}) + "\n")
    p.stdin.flush()
    res = {}

    def rd():
        line = p.stdout.readline()
        if line:
            res.update(json.loads(line))

    th = threading.Thread(target=rd, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive() or not res:
        killed = "killed_gb" in _k
        p.kill()
        if th.is_alive():
            msg = f"[call exceeded {timeout:.0f} s; kernel killed, all variables lost]"
        elif killed:
            msg = f"[kernel killed: memory cap {CAP_GB} GB exceeded ({_k.pop('killed_gb'):.2f} GB); all variables lost]"
        else:
            msg = "[kernel crashed; all variables lost]"
        _k["restarts"] += 1
        _k["p"] = None
        return note + msg, True, timeout
    return note + res["out"], res["error"], res["secs"]


def handle(req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": req["params"].get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "audit", "version": "1.0"}}}
    if m == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": "python", "description": TOOL_DOC,
            "inputSchema": {"type": "object", "properties": {"code": {"type": "string",
                            "description": "Python code to run in the persistent kernel"}}, "required": ["code"]}}]}}
    if m == "tools/call":
        code = req["params"].get("arguments", {}).get("code", "")
        used = time.time() - T0
        _k["n"] += 1
        n = _k["n"]
        over = n > CALLS or used > WALL
        if over and "submit(" not in code:
            out, err, secs = ("[budget used up: only a call that runs submit(...) is accepted now]", True, 0.0)
        else:
            with _lock:
                out, err, secs = kernel_call(code, timeout=max(30.0, min(600.0, WALL + 300 - used)))
        used = time.time() - T0
        flags = [k for k, rx in SECOND_PATH.items() if rx.search(code)]
        LOG.write(json.dumps({"n": n, "t": round(used, 1), "secs": secs, "error": err, "code": code,
                              "out": out[-20000:], "models": [mm for mm in MODELS if mm in code], "flags": flags,
                              "kernel_peak_gb": round(_k.get("peak", 0.0), 2), "restarts": _k["restarts"]}) + "\n")
        LOG.flush()
        if len(out) > 12000:
            out = out[:6000] + f"\n[... {len(out) - 12000} characters cut ...]\n" + out[-6000:]
        footer = f"\n[call {n} of {CALLS}; {used / 60:.1f} of {WALL / 60:.0f} min used]"
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": out + footer}],
                                                         "isError": bool(err)}}
    if rid is None:
        return None
    if m == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {m}"}}


def main():
    threading.Thread(target=watch, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line))
        except Exception as e:  # noqa: BLE001
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": repr(e)}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
