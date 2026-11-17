"""Serial driver for the KL-off live arm (one heavy job at a time on the mini).

For each seed: live.py with --kl-prefix 0 (thought KL off; everything else = live_k4 defaults), run in process
segments that end right after the reads at --seg, 2*--seg, ... (memory creep, parent DEVIATIONS D6). A segment
resumes from adapter_<N>.pt + opt_<N>.pt of its last read. If the watchdog kills a segment, the updates after the
last saved read are moved aside (steps_killed_*.jsonl, batches_killed/) and the run resumes from that read.
A seed ends at --steps, at the frozen onset stop (onset_stop.json), or at the clock (clock_stop.json); the clock
stop ends the whole driver.

  /usr/bin/python3 run_seeds.py --seeds 0 1 2 --stop-at 20:00
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = os.environ.get("LIVE_PY", str(HERE / "../../../env/.venv/bin/python"))


def last_read(rd):
    p = rd / "reads.jsonl"
    if not p.exists():
        return None
    st = [json.loads(l)["step"] for l in p.read_text().splitlines() if l.strip()]
    return st[-1] if st else None


def trim_after(rd, n, tag):
    """Keep steps <= n (a killed segment wrote past its last saved read)."""
    keep, kill = [], []
    for l in (rd / "steps.jsonl").read_text().splitlines():
        try:
            (keep if json.loads(l)["step"] <= n else kill).append(l)
        except Exception:
            pass  # a line cut by the kill
    (rd / "steps.jsonl").write_text("".join(x + "\n" for x in keep))
    (rd / f"steps_killed_{tag}.jsonl").write_text("".join(x + "\n" for x in kill))
    kd = rd / "batches_killed"
    kd.mkdir(exist_ok=True)
    for f in glob.glob(str(rd / "batches" / "*.npz")):
        if int(f.split("_")[-1][:5]) > n:
            shutil.move(f, kd)
    return len(keep), len(kill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seg", type=int, default=300)
    ap.add_argument("--stop-at", default="")
    ap.add_argument("--stop-onset", type=float, default=0.30)
    ap.add_argument("--stop-confirm", type=int, default=2)
    ap.add_argument("--cap-gb", type=float, default=9.0)
    ap.add_argument("--prefix", default="kl0_s")
    ap.add_argument("--max-kills", type=int, default=3)
    ap.add_argument("--read-every", type=int, default=100, help="smoke only")
    ap.add_argument("--cue-delta", type=float, default=0.10, help="0.10 = the status arm; 0.86 = positive control")
    a = ap.parse_args()
    os.chdir(HERE)
    (HERE / "logs").mkdir(exist_ok=True)
    for seed in a.seeds:
        name = f"{a.prefix}{seed}"
        rd = HERE / "data" / "runs" / name
        kills = 0
        while True:
            n = last_read(rd) or 0
            if (rd / "onset_stop.json").exists() or n >= a.steps:
                print(f"[driver] {name} finished at read {n} {time.strftime('%T')}", flush=True)
                break
            if (rd / "clock_stop.json").exists():
                print(f"[driver] clock stop in {name} at {n}; driver ends {time.strftime('%T')}", flush=True)
                return
            end = min((n // a.seg + 1) * a.seg, a.steps)
            cmd = [PY, "live.py", "--name", name, "--seed", str(seed), "--kl-prefix", "0", "--k-stale", "4",
                   "--kl-rows", "16", "--chunk", "4", "--steps", str(a.steps),
                   "--stop-onset", str(a.stop_onset), "--stop-confirm", str(a.stop_confirm),
                   "--read-every", str(a.read_every), "--cue-delta", str(a.cue_delta)]
            if a.stop_at:
                cmd += ["--stop-at", a.stop_at]
            if n:
                cmd += ["--resume", str(n)]
            if end < a.steps:
                cmd += ["--exit-after", str(end)]
            log = HERE / "logs" / f"{name}_from{n}.log"
            print(f"[driver] {name} segment {n}->{end} start {time.strftime('%T')} log {log.name}", flush=True)
            rc = subprocess.call([PY, "watchdog.py", "--cap-gb", str(a.cap_gb), "--log", str(log), "--"] + cmd)
            n2 = last_read(rd) or 0
            print(f"[driver] {name} segment exit {rc}, last read {n2} {time.strftime('%T')}", flush=True)
            if rc != 0:
                kills += 1
                kept, gone = trim_after(rd, n2, f"from{n}_kill{kills}")
                print(f"[driver] {name} KILLED/FAILED (rc {rc}); resume from read {n2}; kept {kept} steps, moved "
                      f"{gone}", flush=True)
                if kills > a.max_kills or n2 == n and kills > 1:
                    print(f"[driver] {name} gave up after {kills} failures", flush=True)
                    break


if __name__ == "__main__":
    main()
