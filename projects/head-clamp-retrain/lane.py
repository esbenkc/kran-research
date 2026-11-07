"""One-job-at-a-time lane: runs the lines of queue.txt in order, each under watchdog.py (8 GB cap, bound to the
child pid). A line is '<name> <script> <args...>'; it runs once (marker logs/<name>.done holds the return code).
queue.txt is re-read before every job, so lines can be appended while the lane runs; a line 'STOP' ends the lane.
Never edit a line that has started. Log: logs/lane.log ('start <name>' / 'end <name> rc=<rc> <secs>s').

    $PY lane.py      (run as a named service; it polls queue.txt every 20 s when idle)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
LOG = HERE / "logs" / "lane.log"


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def main() -> None:
    log("lane up")
    while True:
        todo = None
        for line in (HERE / "queue.txt").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "STOP":
                log("STOP")
                return
            name, *cmd = line.split()
            if not (HERE / "logs" / f"{name}.done").exists():
                todo = (name, cmd)
                break
        if todo is None:
            time.sleep(20)
            continue
        name, cmd = todo
        log(f"start {name}: {' '.join(cmd)}")
        t0 = time.time()
        rc = subprocess.call([PY, str(HERE / "watchdog.py"), "--cap-gb", "8.0", "--log", str(HERE / "logs" / f"{name}.log"),
                              "--", PY, str(HERE / cmd[0]), *cmd[1:]], cwd=HERE)
        (HERE / "logs" / f"{name}.done").write_text(f"rc={rc}\n")
        log(f"end {name} rc={rc} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
