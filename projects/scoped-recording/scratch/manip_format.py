"""Scratch (smoke seeds only): does a format line after the frozen M1/M2 text fix the unparsed replies?
Variant F appends FORMAT after the question; the question text itself is unchanged."""
import json, glob, sys, collections
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import concurrent.futures as cf
import manip, agent
FORMAT = "Reply with that one word only. Do not write JSON and do not call a tool."
jobs = []
for qid, q in manip.QUESTIONS.items():
    for arm in q["arms"]:
        for p in sorted(glob.glob(f"raw_smoke/*/{arm}/*.jsonl")):
            jobs.append((p, qid))
def one(j):
    p, qid = j
    rec = json.loads(open(p).read()); q = manip.QUESTIONS[qid]
    prompt, _ = manip.replay_prompt(rec, q["text"] + "\n" + FORMAT)
    t, meta = agent.complete_cached(rec["model"], prompt, agent.system_prompt(rec["cell"]))
    a = manip.parse(t, q["options"])
    return qid, rec["model"], rec["cell"], a, (a == q["correct"].get(rec["cell"])) if qid == "M1" else None, t[:80]
with cf.ThreadPoolExecutor(12) as ex:
    rows = list(ex.map(one, jobs))
agg = collections.defaultdict(lambda: [0, 0, 0])
for qid, m, arm, a, c, t in rows:
    k = (qid, m[:14]); agg[k][0] += 1; agg[k][1] += a is None; agg[k][2] += bool(c)
    print(qid, m[:14], arm, a, repr(t))
for k, v in sorted(agg.items()): print(k, "n", v[0], "unparsed", v[1], "correct", v[2])
