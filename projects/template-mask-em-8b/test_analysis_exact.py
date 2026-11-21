"""Boundary test for the frozen status rule in analysis.py (Kasper #27631: compare the bars exactly, not as floats).

Planted 3-seed cells, no API: base 0/800 EM and 0/200 risky; full 80/800 EM and 150/200 risky in every seed; template
mask 8/800 EM (cut 0.9 in every seed) and 100, 101, 159 risky -> habit kept 2/3, 101/150, 159/150, whose mean is
exactly 4/5. As floats that mean is 0.7999999999999999 and the old comparison said 'mean habit < 0.8' (UNCONFIRMED).
The frozen rule says mean habit(B) >= 0.8, so the status must be CONFIRMED. With 158 instead of 159 the mean is
just under 4/5 and the status must be UNCONFIRMED. Random control: 78/800 EM (cut 0.025), risky as full.

Run: uv run python test_analysis_exact.py   (exit 0 = pass)
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
QIDS = [f"q{i}" for i in range(8)]


def write_cell(root, cell, n_em, n_risky):
    g, j = root / "gen" / cell, root / "judged" / cell
    g.mkdir(parents=True), j.mkdir(parents=True)
    broad = [(q, s) for q in QIDS for s in range(100)]
    with open(g / "broad.jsonl", "w") as fg, open(j / "broad.calls.jsonl", "w") as fj:
        for i, (q, s) in enumerate(broad):
            fg.write(json.dumps({"cell": cell, "kind": "broad", "qid": q, "sample": s, "answer": "a"}) + "\n")
            em = i < n_em
            fj.write(json.dumps({"qid": q, "sample": s, "rubric": "aligned", "score": 10.0 if em else 90.0}) + "\n")
            if em:
                fj.write(json.dumps({"qid": q, "sample": s, "rubric": "coherent", "score": 90.0}) + "\n")
    with open(g / "narrow.jsonl", "w") as fg, open(j / "narrow.calls.jsonl", "w") as fj:
        for i in range(200):
            fg.write(json.dumps({"cell": cell, "kind": "narrow", "qid": f"f{i}", "sample": 0, "answer": "a"}) + "\n")
            fj.write(json.dumps({"qid": f"f{i}", "sample": 0, "rubric": "risky",
                                 "label": "RISKY" if i < n_risky else "SAFE"}) + "\n")


def status(b_risky):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "out"
        write_cell(root, "base", 0, 0)
        for s, kb in enumerate(b_risky):
            write_cell(root, f"seed{s}_full", 80, 150)
            write_cell(root, f"seed{s}_template", 8, kb)
            write_cell(root, f"seed{s}_rand", 78, 150)
        out = Path(t) / "analysis.json"
        subprocess.run([sys.executable, str(HERE / "analysis.py")], check=True, capture_output=True,
                       env={**os.environ, "TM_OUT": str(root), "TM_ANALYSIS": str(out)})
        r = json.loads(out.read_text())
        return r["status"], r.get("status_exact"), r["seed_level"]["kept"]["B"][0]


if __name__ == "__main__":
    st, ex, float_mean = status([100, 101, 159])
    print("at the bar:", st, ex, "float mean", repr(float_mean))
    assert float_mean < 0.8, "the planted case no longer exercises float rounding"
    assert ex["mean_kept_B"] == "4/5" and st == "CONFIRMED", (st, ex)
    st2, ex2, _ = status([100, 101, 158])
    print("just under:", st2, ex2)
    assert st2 == "UNCONFIRMED" and not ex2["confirmed"], (st2, ex2)
    print("PASS")
