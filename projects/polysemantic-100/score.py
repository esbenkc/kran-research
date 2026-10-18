"""Steps 3-4: score every hypothesis on the model's own activations, then let 3 judges decide
whether the hypotheses that passed on one pole are RELATED or UNRELATED.

    with-model-keys uv run python projects/polysemantic-100/score.py   -> data/labels.json

Probe rule (fixed before probe results were read):
  For pole sign s of neuron k: f_i = s * (strongest firing of k on fire text i),
  c_i = the same on its control text. ref = max(corpus pole max, all probe texts of this pole).
  A hypothesis PASSES if >= 3 of 4 fire texts reach f_i >= 0.5 * ref AND >= 3 of 4 pairs have
  c_i < 0.5 * f_i.
Pole label:
  TOO_SPARSE  < 5 strong windows in the corpus
  NO_PASS     no hypothesis passed (we could not explain the pole)
  ONE_JOB     exactly 1 hypothesis passed
  RELATED     >= 2 passed and the judge majority says RELATED
  UNRELATED   >= 2 passed and the judge majority says UNRELATED
Neuron label = the "worst" pole label in the order UNRELATED > RELATED > ONE_JOB > NO_PASS > TOO_SPARSE.
Nulls:
  swap   each hypothesis scored on all 99 other neurons (same sign), same rule. Bar: <= 5% pass.
  pooled judge sees 2 passed hypotheses from 2 different neurons. Bar: >= 80% UNRELATED.
  split  judge sees 1 passed hypothesis twice, its snippets split in two halves. Bar: >= 80% RELATED.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA, PROJECT_DIR, load_neurons
from spend import call

JUDGES = ["claude-haiku-4-5", "gemini-3.8-flash", "claude-sonnet-4-6"]  # OpenAI key has no credits (429, 2026-10-17)
JUDGE_PROMPT = (PROJECT_DIR / "prompts" / "judge.md").read_text()
ORDER = ["UNRELATED", "RELATED", "ONE_JOB", "NO_PASS", "TOO_SPARSE"]


def load():
    neurons = load_neurons()
    col = {n["id"]: k for k, n in enumerate(neurons)}
    ctx = {n["id"]: json.loads((DATA / "contexts" / f"{n['id']}.json").read_text()) for n in neurons}
    hyps = json.loads((DATA / "hypotheses.json").read_text())
    acts = json.loads((DATA / "probe_acts.json").read_text())
    return neurons, col, ctx, hyps, acts


def val(acts, key, k, sign):
    r = acts[key]
    return r["max"][k] if sign == "pos" else -r["min"][k]


def probe_pass(acts, r, h_i, k, corpus_max, all_keys):
    s = r["pole"]
    f = np.array([val(acts, f"{r['id']}|{s}|{h_i}|fire|{i}", k, s) for i in range(4)])
    c = np.array([val(acts, f"{r['id']}|{s}|{h_i}|control|{i}", k, s) for i in range(4)])
    ref = max(corpus_max, max(val(acts, key, k, s) for key in all_keys))
    ok = bool((f >= 0.5 * ref).sum() >= 3 and (c < 0.5 * np.maximum(f, 1e-9)).sum() >= 3)
    return ok, {"fire": np.round(f / ref, 3).tolist(), "control": np.round(c / ref, 3).tolist(),
                "ref": round(ref, 4), "corpus_max": corpus_max, "probe_over_corpus": round(ref / corpus_max, 2)}


def describe(h, pole, snippets=None):
    ex = snippets if snippets is not None else [pole["examples"][i - 1] for i in h.get("covers", [])
                                                if 1 <= i <= len(pole["examples"])][:3]
    lines = [f"* {h['name']} ({h['kind']}): {h['description']}"]
    lines += ["    snippet: " + e["text"].replace("\n", "\\n") for e in ex]
    return "\n".join(lines)


def judge(block: str, tag: str) -> dict:
    votes = []
    for m in JUDGES:
        t = call(JUDGE_PROMPT + block, model=m, max_tokens=400, tag=tag)
        mm = re.search(r'"verdict"\s*:\s*"(RELATED|UNRELATED)"', t)
        votes.append({"model": m, "verdict": mm.group(1) if mm else "PARSE_FAIL", "raw": t[:300]})
    n_un = sum(v["verdict"] == "UNRELATED" for v in votes)
    return {"verdict": "UNRELATED" if n_un >= 2 else "RELATED", "votes": votes}


def main() -> None:
    neurons, col, ctx, hyps, acts = load()
    ids = [n["id"] for n in neurons]
    poles, passed_all = [], []
    swap_n = swap_pass = 0
    for r in hyps:
        pole = ctx[r["id"]][r["pole"]]
        rec = {"id": r["id"], "pole": r["pole"], "n_strong_windows": pole["n_strong_windows"],
               "pole_max": pole["pole_max"], "hyps": []}
        if r["status"] == "TOO_SPARSE":
            rec["label"] = "TOO_SPARSE"
            poles.append(rec)
            continue
        for h_i, h in enumerate(r["hypotheses"]):
            keys = [f"{r['id']}|{r['pole']}|{h_i}|{kind}|{i}" for kind in ("fire", "control") for i in range(4)]
            ok, nums = probe_pass(acts, r, h_i, col[r["id"]], pole["pole_max"], keys)
            rec["hyps"].append({**{k: h[k] for k in ("name", "kind", "description", "covers")},
                                "pass": ok, **nums})
            if ok:
                passed_all.append((r["id"], r["pole"], h, pole))
            for other in ids:  # swap null
                if other == r["id"]:
                    continue
                om = ctx[other][r["pole"]]["pole_max"]
                if om <= 0:
                    continue
                swap_n += 1
                swap_pass += probe_pass(acts, r, h_i, col[other], om, keys)[0]
        passed = [(h, rh) for h, rh in zip(r["hypotheses"], rec["hyps"]) if rh["pass"]]
        if not passed:
            rec["label"] = "NO_PASS"
        elif len(passed) == 1:
            rec["label"] = "ONE_JOB"
        else:
            rec["judge_block"] = "\n".join(describe(h, pole) for h, _ in passed)
            rec["label"] = None
        poles.append(rec)

    todo = [p for p in poles if p["label"] is None]
    with ThreadPoolExecutor(6) as ex:
        verdicts = list(ex.map(lambda p: judge(p["judge_block"], "judge"), todo))
    for p, v in zip(todo, verdicts):
        p["label"], p["judge"] = v["verdict"], v

    # Judge nulls.
    rng = np.random.default_rng(1)
    pooled, split = [], []
    for _ in range(min(30, len(passed_all))):
        a, b = rng.choice(len(passed_all), 2, replace=False)
        if passed_all[a][0] == passed_all[b][0]:
            continue
        pooled.append("\n".join(describe(x[2], x[3]) for x in (passed_all[a], passed_all[b])))
    for nid, sign, h, pole in passed_all:
        cov = [pole["examples"][i - 1] for i in h.get("covers", []) if 1 <= i <= len(pole["examples"])]
        if len(cov) >= 4 and len(split) < 30:
            half = len(cov) // 2
            split.append(describe(h, pole, cov[:half][:3]) + "\n" + describe(h, pole, cov[half:][:3]))
    with ThreadPoolExecutor(6) as ex:
        pv = list(ex.map(lambda b: judge(b, "null-pooled"), pooled))
        sv = list(ex.map(lambda b: judge(b, "null-split"), split))

    by_neuron = {}
    for p in poles:
        by_neuron.setdefault(p["id"], []).append(p["label"])
    neuron_label = {i: min(ls, key=ORDER.index) for i, ls in by_neuron.items()}
    pole_votes = [p["judge"]["votes"] for p in poles if p.get("judge")]
    out = {
        "n_neurons": len(neuron_label), "n_poles": len(poles),
        "neuron_labels": Counter(neuron_label.values()),
        "pole_labels": Counter(p["label"] for p in poles),
        "null_swap": {"n": swap_n, "pass": swap_pass, "rate": round(swap_pass / max(swap_n, 1), 4)},
        "null_pooled": {"n": len(pv), "unrelated": sum(v["verdict"] == "UNRELATED" for v in pv)},
        "null_split": {"n": len(sv), "related": sum(v["verdict"] == "RELATED" for v in sv)},
        "judge_unanimous": sum(len({v["verdict"] for v in vs}) == 1 for vs in pole_votes),
        "judge_n": len(pole_votes),
        "neurons": neuron_label, "poles": poles,
        "nulls_raw": {"pooled": [(b, v) for b, v in zip(pooled, pv)], "split": [(b, v) for b, v in zip(split, sv)]},
    }
    (DATA / "labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k in ("neuron_labels", "pole_labels", "null_swap", "null_pooled", "null_split"):
        print(k, out[k])
    print("judge unanimous", out["judge_unanimous"], "/", out["judge_n"])


if __name__ == "__main__":
    main()
