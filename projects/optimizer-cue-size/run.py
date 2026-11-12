"""optimizer-cue-size: rebuild every number of the post from the saved runs (seconds), or retrain (hours).

    /usr/bin/python3 run.py            # pick_lr.py (3 arms) + analyze.py + mech.py from data/runs/*.json
    ./lane.sh A queue.txt              # retrain every run in queue order (torch MPS venv, see README.md)

The figures are built in the blog clone from data/figure_data.json (figure1/main_data.py, figure1/fig1.py).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
for cmd in (["pick_lr.py", "--arm", "sgd"], ["pick_lr.py", "--arm", "adamwtm"], ["pick_lr.py", "--arm", "sgdfa"],
            ["analyze.py"], ["mech.py"]):
    print("$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)
