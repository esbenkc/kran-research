"""PREREG v1.3 s6 cap rule from the smoke: s = mean seconds per step with the in-loop refits included;
cap = the largest multiple of 50 in [300, 900] with 2 x (cap x s + 600) <= seconds from now to 10:50 PDT.
s = train-step seconds (steps 2..12 of the smoke) + (check-0 seconds x n_check_prompts_full / n_check_prompts_smoke) / 50
(the smoke checks 400 prompts; a full check captures 640 fit + 800 eval = 1440; capture dominates the check;
check-0 seconds include the model load, so s is slightly high = conservative).
Usage: venv/bin/python cap.py data/runs/smoke_lie_spread__lam10000__seed99.json [--now HH:MM]
"""
import datetime as dt
import json
import sys

r = json.loads(open(sys.argv[1]).read())
log, traj = r["log"], r["traj"]
step_s = (log[-1]["sec"] - log[1]["sec"]) / (len(log) - 2) if len(log) > 2 else log[-1]["sec"] / len(log)
check_s = traj[0]["sec"]
full_check = check_s * 1440 / (r["n_fit_prompts"] + 200)  # smoke: 200 fit + 200 eval prompts
s = step_s + full_check / 50
now = dt.datetime.now()
if "--now" in sys.argv:
    hh, mm = sys.argv[sys.argv.index("--now") + 1].split(":")
    now = now.replace(hour=int(hh), minute=int(mm))
left = (now.replace(hour=10, minute=50, second=0) - now).total_seconds()
cap = None
for c in range(900, 299, -50):
    if 2 * (c * s + 600) <= left:
        cap = c
        break
out = {"train_step_s": round(step_s, 2), "check_s_smoke": round(check_s, 1), "check_s_full_est": round(full_check, 1),
       "s": round(s, 2), "seconds_left_to_10_50": round(left), "cap": cap, "now": now.strftime("%H:%M:%S")}
print(json.dumps(out))
