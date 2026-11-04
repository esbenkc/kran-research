"""Iris: every 'denied: approved token required' reply in raw_smoke, checked against the true approved token and
the task's op field (env.make_state, env.TASKS). Usage: python firstop_token_check.py <project dir>"""
import json, glob, collections, sys
proj = sys.argv[1] if len(sys.argv) > 1 else "kran-research/projects/residue-as-trait"
sys.path.insert(0, proj)
import env
c = collections.Counter(); fields = collections.Counter()
for f in glob.glob(f"{proj}/raw_smoke/*/*/*/*.jsonl"):
    task, model, cell, seed = f.split("/")[-4:]; seed = int(seed[:-6])
    st, T = env.make_state(seed, task), env.TASKS[task]
    r = json.loads(open(f).read().splitlines()[-1])
    for a, o in zip(r["actions"], r["observations"][1:]):
        if "approved token required" in o:
            b = (a.get("args") or {}).get("body") or {}
            c[(str(b.get("token", "")) == st.scoped_token, str(b.get(T.op_field)) == T.op_value)] += 1
            fields[(task, tuple(sorted(k for k in b if k != "token")))] += 1
print("(right_token, right_field):", dict(c))
for k, v in fields.most_common(12): print(v, k)
