"""Appendix: Tanya's follow-up questions on the elegance post.

    uv run python projects/proof-elegance/appendix.py [--tight]

1. Does any single rubric score differ between AI and human proofs?  -> figure `elegance-dimensions`
2. Can a classifier tell AI math from human math?                    -> leave-one-out accuracy
3. How "optimized" are the proofs?  (--tight runs the judges)        -> share of each proof that
   the judges say could be cut without losing the argument (ProofRank-style conciseness).

Primary corpus only (36 AI / 36 human). Writes appendix_results.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib import figure
import importlib.util

P = Path(__file__).parent
DATA = P / "data"
TIGHT = DATA / "scores_tight"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, P / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load_module("run")      # load(), compare(), DIMS
score = _load_module("score")  # judges + text_of()

DIM_LABEL = {"economy": "Economy", "surprise": "Surprising key idea", "explanation": "Explanatory power",
             "simplicity": "Simple structure", "conceptual": "Conceptual, not computational",
             "unification": "Cross-field unification"}

# Placeholders the blinding step inserts only into human texts would leak the label.
PLACEHOLDER = re.compile(r"\[author\]|\[year\]|\[\.\.\.\]|\[\d+(?:,\s*\d+)*\]")
RATE_WORDS = ["we", "our", "note", "remark", "clearly", "consequently", "therefore", "hence", "thus"]


def prose(text: str) -> str:
    """English prose only: drop math, LaTeX commands and blinding placeholders, so the classifier
    cannot key on macro habits (\\leqslant vs \\le) or on placeholders only human texts contain."""
    t = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return PLACEHOLDER.sub(" ", t)


def classifier(items: list[dict], corpus: dict) -> dict:
    y = np.array([r["source"] == "ai" for r in items], dtype=int)
    loo = LeaveOneOut()
    # (a) from the six rubric scores alone: does elegance itself give the author away?
    X = np.array([[r[d] for d in run.DIMS] for r in items])
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=loo)
    # (b) from the words of the prose (math and LaTeX removed)
    texts = [prose((P / corpus[r["id"]]["path"]).read_text()) for r in items]
    mk = lambda: make_pipeline(  # noqa: E731
        TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", min_df=4, sublinear_tf=True),
        LogisticRegression(max_iter=4000, C=10))
    pred_w = cross_val_predict(mk(), texts, y, cv=loo)
    pipe = mk().fit(texts, y)
    names = pipe[0].get_feature_names_out()
    order = np.argsort(pipe[1].coef_[0])
    words = {s: sum(len(t.split()) for t, r in zip(texts, items) if r["source"] == s) for s in ("ai", "human")}
    rates = {w: {s: 1000 * sum(len(re.findall(rf"\b{w}\b", t, re.I)) for t, r in zip(texts, items) if r["source"] == s)
                 / words[s] for s in ("ai", "human")} for w in RATE_WORDS}
    return {"n": len(y), "chance": 0.5,
            "rubric_scores_loo_acc": float((pred == y).mean()), "prose_words_loo_acc": float((pred_w == y).mean()),
            "tells": {"ai": list(names[order[::-1][:30]]), "human": list(names[order[:30]])},
            "per_1000_words": rates}


TIGHT_PROMPT = """You are an experienced research mathematician and a strict editor.
Below is the proof of a research result (an excerpt: `[...]` marks omitted passages, `[n]` are citations).
Correctness is out of scope; assume it is correct.

Question: how much of this text could be deleted or compressed WITHOUT losing any step of the argument?
Count redundancy, restated facts, over-detailed routine verification, and lemmas stated at an unnatural
level of generality. Do not count steps that are merely hard. Do not propose a different proof.

Return ONLY a JSON object:
{{"cuttable_percent": <0-100>, "biggest_cut": "<one sentence: the largest thing you would cut>",
  "better_idea_likely": "yes" | "no" | "unsure"}}

"better_idea_likely" = whether you think a substantially shorter proof by a different idea probably exists.

The proof:
<<<
{proof}
>>>"""


def tight_one(judge: str, item: dict, text: str) -> str:
    out = TIGHT / judge / f"{item['id']}.json"
    if out.exists():
        return "cached"
    last = None
    for _ in range(6):
        try:
            raw, model = score.JUDGES[judge](TIGHT_PROMPT.format(proof=text))
            m = re.search(r"\{.*\}", raw, re.S)
            res = json.loads(m.group(0))
            if not 0 <= float(res["cuttable_percent"]) <= 100:
                raise ValueError("out of range")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"id": item["id"], "judge": judge, "model": model, **res}, indent=1))
            return "ok"
        except Exception as e:
            last = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                import time
                time.sleep(60)
    return f"failed: {last}"


def run_tight(items: list[dict], corpus: dict) -> None:
    import concurrent.futures as cf
    jobs = [(j, r) for j in score.JUDGES for r in items]
    with cf.ThreadPoolExecutor(4) as ex:
        for (j, r), st in zip(jobs, ex.map(lambda jr: tight_one(jr[0], jr[1], (P / corpus[jr[1]["id"]]["path"]).read_text()), jobs)):
            if st != "cached":
                print(j, r["id"], st)


def tightness(items: list[dict]) -> dict | None:
    rows = []
    for r in items:
        got = {j: json.loads((TIGHT / j / f"{r['id']}.json").read_text()) for j in score.JUDGES
               if (TIGHT / j / f"{r['id']}.json").exists()}
        if len(got) == len(score.JUDGES):
            rows.append({"source": r["source"], "cut": float(np.mean([g["cuttable_percent"] for g in got.values()])),
                         "better": sum(g["better_idea_likely"] == "yes" for g in got.values()),
                         "cuts": {j: g["biggest_cut"] for j, g in got.items()}, "problem": r["problem"]})
    if not rows:
        return None
    a = np.array([x["cut"] for x in rows if x["source"] == "ai"])
    h = np.array([x["cut"] for x in rows if x["source"] == "human"])
    ab = [x["better"] for x in rows if x["source"] == "ai"]
    hb = [x["better"] for x in rows if x["source"] == "human"]
    return {"n_ai": len(a), "n_human": len(h), "cut_mean_ai": float(a.mean()), "cut_mean_human": float(h.mean()),
            "cut_median_ai": float(np.median(a)), "cut_median_human": float(np.median(h)),
            "mannwhitney_p": float(stats.mannwhitneyu(a, h).pvalue),
            "better_idea_both_judges_ai": sum(b == 2 for b in ab), "better_idea_both_judges_human": sum(b == 2 for b in hb),
            "most_cuttable": sorted(rows, key=lambda x: -x["cut"])[:4]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tight", action="store_true", help="run the tightness judges (cached)")
    args = ap.parse_args()
    items, _ = run.load()
    corpus = {r["id"]: r for r in map(json.loads, (DATA / "corpus.jsonl").read_text().splitlines()) if r}
    if args.tight:
        run_tight(items, corpus)

    res = {"dimensions": {d: run.compare(items, d) for d in run.DIMS},
           "classifier": classifier(items, corpus),
           "tightness": tightness(items)}
    (P / "appendix_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({d: {k: round(v, 3) if isinstance(v, float) else v for k, v in res["dimensions"][d].items()
                          if k in ("mean_ai", "mean_human", "mannwhitney_p")} for d in run.DIMS}, indent=0))
    c = res["classifier"]
    print("classifier:", {k: c[k] for k in ("rubric_scores_loo_acc", "prose_words_loo_acc")})
    print("tells ai:", c["tells"]["ai"][:20]); print("tells human:", c["tells"]["human"][:20])
    print("rates /1000 words:", {w: {s: round(v, 2) for s, v in r.items()} for w, r in c["per_1000_words"].items()})
    if res["tightness"]:
        t = res["tightness"]
        print("tightness:", {k: v for k, v in t.items() if k != "most_cuttable"})

    fig = {"dims": [{"key": d, "label": DIM_LABEL[d]} for d in run.DIMS],
           "items": [{"source": r["source"], "problem": r["problem"] or r["title"], "url": r["url"],
                      **{d: round(r[d], 2) for d in run.DIMS}} for r in items],
           "summary": {d: {"ai": res["dimensions"][d]["mean_ai"], "human": res["dimensions"][d]["mean_human"],
                           "p": res["dimensions"][d]["mannwhitney_p"]} for d in run.DIMS}}
    plot = (P / "plot-dimensions.js").read_text() if (P / "plot-dimensions.js").exists() else None
    figure.save(P, "elegance-dimensions", fig, plot)


if __name__ == "__main__":
    main()
