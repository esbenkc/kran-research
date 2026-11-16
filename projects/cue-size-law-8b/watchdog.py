"""Run a command, poll its phys_footprint (macOS `footprint <pid>`), kill it above a cap.

    .venv/bin/python projects/probe-sensing-mps/watchdog.py --cap-gb 5.0 --log x.log -- <cmd...>
Bound to the child's own pid (no pgrep), so it cannot outlive its job or kill another run.
"""
import argparse
import re
import subprocess
import sys
import time

UNIT = {"B": 1 / 2**30, "KB": 1 / 2**20, "MB": 1 / 2**10, "GB": 1.0}


def footprint_gb(pid: int) -> float | None:
    try:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", out)
    return float(m.group(1)) * UNIT[m.group(2)] if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, default=5.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    with open(a.log, "a") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        peak = 0.0
        while p.poll() is None:
            gb = footprint_gb(p.pid)
            if gb is not None:
                peak = max(peak, gb)
                if gb > a.cap_gb:
                    log.write(f"\nWATCHDOG: footprint {gb:.2f} GB > cap {a.cap_gb} GB, killing {p.pid}\n")
                    log.flush()
                    p.kill()
                    break
            time.sleep(3)
        rc = p.wait()
        log.write(f"\nWATCHDOG: exit {rc}, peak footprint {peak:.2f} GB\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
