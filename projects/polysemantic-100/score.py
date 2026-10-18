"""Steps 3-4: score every hypothesis on the model's own activations, then let 3 judges decide
whether the hypotheses that passed on one pole are RELATED or UNRELATED.

    with-model-keys uv run --with mlx-lm python projects/polysemantic-100/score.py   -> data/labels.json

Probe rule (fixed before probe results were read):
  For pole sign s of neuron k: f_i = s * (strongest firing of k on fire text i),
  c_i = the same on its control text. A hypothesis PASSES if >= 3 of 4 fire texts reach
  f_i >= 0.5 * ref AND >= 3 of 4 pairs have c_i < 0.5 * f_i.
  PRIMARY ref     = max(corpus pole max, all probe texts of ALL hypotheses of this pole).
  SENSITIVITY ref = max(corpus pole max, the 8 texts of this hypothesis only) (reported, not used
  for the labels).
Pole label:
  TOO_SPARSE  < 5 strong windows in the corpus
  NO_PASS     no hypothesis passed (we could not explain the pole)
  ONE_JOB     exactly 1 hypothesis passed (failed hypotheses may hide a 2nd job: lower bound)
  RELATED     >= 2 passed and the judge majority says RELATED
  UNRELATED   >= 2 passed and the judge majority says UNRELATED
Neuron label = the "worst" pole label in the order UNRELATED > RELATED > ONE_JOB > NO_PASS > TOO_SPARSE.
Nulls:
  swap   each hypothesis scored on all 99 other neurons (same sign), same rule. Bar: <= 5% pass.
  pooled judge sees 2 passed hypotheses from 2 different neurons. Bar: >= 80% UNRELATED.
  split  for passed hypotheses with >= 6 covered snippets: split the snippets in two halves, run
         the explain prompt on each half separately, judge the two independent descriptions.
         Bar: >= 80% RELATED.
Sensitivity (reported, not relabeled): UNRELATED poles where two passed jobs have the same most
common firing token string on their fire texts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA, PROJECT_DIR, load_neurons
from explain import MODEL as EXPLAIN_MODEL, PROMPT as EXPLAIN_PROMPT, parse, snippet_block
from spend import call

JUDGES = ["claude-haiku-4-5", "gemini-3.8-flash", "claude-sonnet-4-6"]  # OpenAI key has no credits (429, 2026-10-17)
JUDGE_PROMPT = (PROJECT_DIR / "prompts" / "judge.md").read_text()
ORDER = ["UNRELATED", "RELATED", "ONE_JOB", "NO_PASS", "TOO_SPARSE"]


def val(acts, key, k, sign):
    r = acts[key]
    return r["max"][k] if sign == "pos" else -r["min"][k]


def keys_of(nid, sign, h_i):
    return [f"{nid}|{sign}|{h_i}|{kind}|{i}" for kind in ("fire", "control") for i in range(4)]


def probe_pass(acts, nid, sign, h_i, k, corpus_max, ref_keys):
    f = np.array([val(acts, f"{nid}|{sign}|{h_i}|fire|{i}", k, sign) for i in range(4)])
    c = np.array([val(acts, f"{nid}|{sign}|{h_i}|control|{i}", k, sign) for i in range(4)])
    ref = max(corpus_max, max(val(acts, key, k, sign) for key in ref_keys))
    ok = bool((f >= 0.5 * ref).sum() >= 3 and (c < 0.5 * np.maximum(f, 1e-9)).sum() >= 3)
    return ok, {"fire": np.round(f / ref, 3).tolist(), "control": np.round(c / ref, 3).tolist(),
                "ref": round(ref, 4), "probe_over_corpus": round(ref / corpus_max, 2)}


def describe(h, examples):
    lines = [f"* {h['name']} ({h['kind']}): {h['description']}"]
    lines += ["    snippet: " + e["text"].replace("\n", "\\n") for e in examples[:3]]
    return "\n".join(lines)


def covered(h, pole):
    return [pole["examples"][i - 1] for i in h.get("covers", []) if isinstance(i, int) and 1 <= i <= len(pole["examples"])]


def judge(block: str, tag: str) -> dict:
    votes = []
    for m in JUDGES:
        t = call(JUDGE_PROMPT + block, model=m, max_tokens=400, tag=tag)
        mm = re.search(r'"verdict"\s*:\s*"(RELATED|UNRELATED)"', t)
        votes.append({"model": m, "verdict": mm.group(1) if mm else "PARSE_FAIL", "raw": t[:300]})
    n_un = sum(v["verdict"] == "UNRELATED" for v in votes)
    return {"verdict": "UNRELATED" if n_un >= 2 else "RELATED", "votes": votes}


def split_block(h, pole):
    """Two independent descriptions, one per half of the hypothesis's covered snippets."""
    cov = covered(h, pole)
    halves = [cov[: len(cov) // 2], cov[len(cov) // 2 :]]
    descs = []
    for half in halves:
        out = parse(call(EXPLAIN_PROMPT + snippet_block({"examples": half}), model=EXPLAIN_MODEL,
                         max_tokens=3000, tag="null-split-explain"))
        hs = out.get("hypotheses") or []
        if not hs:
            return None
        descs.append(describe({**{"name": "", "kind": "", "description": ""}, **hs[0]}, half))
    return "\n".join(descs)


def main() -> None:
    neurons = load_neurons()
    col = {n["id"]: k for k, n in enumerate(neurons)}
    ids = [n["id"] for n in neurons]
    ctx = {i: json.loads((DATA / "contexts" / f"{i}.json").read_text()) for i in ids}
    hyps = json.loads((DATA / "hypotheses.json").read_text())
    acts = json.loads((DATA / "probe_acts.json").read_text())

    poles, passed_all = [], []
    swap_n = swap_pass = 0
    for r in hyps:
        nid, s = r["id"], r["pole"]
        pole = ctx[nid][s]
        rec = {"id": nid, "pole": s, "n_strong_windows": pole["n_strong_windows"],
               "pole_max": pole["pole_max"], "hyps": []}
        if r["status"] == "TOO_SPARSE":
            rec["label"] = rec["label_sens"] = "TOO_SPARSE"
            poles.append(rec)
            continue
        pole_keys = [key for h_i in range(len(r["hypotheses"])) for key in keys_of(nid, s, h_i)]
        for h_i, h in enumerate(r["hypotheses"]):
            ok, nums = probe_pass(acts, nid, s, h_i, col[nid], pole["pole_max"], pole_keys)
            ok_s, nums_s = probe_pass(acts, nid, s, h_i, col[nid], pole["pole_max"], keys_of(nid, s, h_i))
            tok_field = "argmax_tok" if s == "pos" else "argmin_tok"
            top_tok = Counter(acts[f"{nid}|{s}|{h_i}|fire|{i}"][tok_field][col[nid]] for i in range(4)).most_common(1)[0][0]
            rec["hyps"].append({**{k: h.get(k) for k in ("name", "kind", "description", "covers")},
                                "pass": ok, "pass_sens": ok_s, **nums, "ref_sens": nums_s["ref"],
                                "fire_top_token": top_tok})
            if ok:
                passed_all.append((nid, s, h, pole))
            for other in ids:  # swap null: this hypothesis's texts scored on every other neuron
                if other == nid or ctx[other][s]["pole_max"] <= 0:
                    continue
                swap_n += 1
                swap_pass += probe_pass(acts, nid, s, h_i, col[other], ctx[other][s]["pole_max"], pole_keys)[0]
        for field, lab in (("pass", "label"), ("pass_sens", "label_sens")):
            passed = [(h, rh) for h, rh in zip(r["hypotheses"], rec["hyps"]) if rh[field]]
            if not passed:
                rec[lab] = "NO_PASS"
            elif len(passed) == 1:
                rec[lab] = "ONE_JOB"
            else:
                rec[lab] = None
                rec[f"block_{lab}"] = "\n".join(describe(h, covered(h, pole)) for h, _ in passed)
        poles.append(rec)

    todo = [(p, lab) for p in poles for lab in ("label", "label_sens") if p[lab] is None]
    with ThreadPoolExecutor(6) as ex:
        verdicts = list(ex.map(lambda t: judge(t[0][f"block_{t[1]}"], "judge"), todo))
    for (p, lab), v in zip(todo, verdicts):
        p[lab] = v["verdict"]
        p[f"judge_{lab}"] = v

    # Judge nulls.
    rng = np.random.default_rng(1)
    pooled = []
    for _ in range(40):
        if len(passed_all) < 2 or len(pooled) >= 30:
            break
        a, b = rng.choice(len(passed_all), 2, replace=False)
        if passed_all[a][0] != passed_all[b][0]:
            pooled.append("\n".join(describe(x[2], covered(x[2], x[3])) for x in (passed_all[a], passed_all[b])))
    split_src = [(h, pole) for _, _, h, pole in passed_all if len(covered(h, pole)) >= 6][:30]
    with ThreadPoolExecutor(6) as ex:
        split = [b for b in ex.map(lambda t: split_block(*t), split_src) if b]
        pv = list(ex.map(lambda b: judge(b, "null-pooled"), pooled))
        sv = list(ex.map(lambda b: judge(b, "null-split"), split))

    def neuron_labels(lab):
        by = {}
        for p in poles:
            by.setdefault(p["id"], []).append(p[lab])
        return {i: min(ls, key=ORDER.index) for i, ls in by.items()}

    nl, nl_s = neuron_labels("label"), neuron_labels("label_sens")
    votes = [p["judge_label"]["votes"] for p in poles if p.get("judge_label")]
    same_tok = [f"{p['id']}|{p['pole']}" for p in poles if p["label"] == "UNRELATED"
                and len({h["fire_top_token"].strip().lower() for h in p["hyps"] if h["pass"]})
                < sum(h["pass"] for h in p["hyps"])]
    out = {
        "n_neurons": len(nl), "n_poles": len(poles),
        "neuron_labels": Counter(nl.values()), "pole_labels": Counter(p["label"] for p in poles),
        "neuron_labels_sens": Counter(nl_s.values()), "pole_labels_sens": Counter(p["label_sens"] for p in poles),
        "null_swap": {"n": swap_n, "pass": swap_pass, "rate": round(swap_pass / max(swap_n, 1), 4)},
        "null_pooled": {"n": len(pv), "unrelated": sum(v["verdict"] == "UNRELATED" for v in pv)},
        "null_split": {"n": len(sv), "related": sum(v["verdict"] == "RELATED" for v in sv)},
        "judge_unanimous": sum(len({v["verdict"] for v in vs}) == 1 for vs in votes), "judge_n": len(votes),
        "unrelated_same_top_token": same_tok,
        "poles_probe_over_corpus_gt2": sum(any(h["probe_over_corpus"] > 2 for h in p["hyps"]) for p in poles),
        "neurons": nl, "neurons_sens": nl_s, "poles": poles,
        "nulls_raw": {"pooled": list(zip(pooled, pv)), "split": list(zip(split, sv))},
    }
    (DATA / "labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k in ("neuron_labels", "pole_labels", "neuron_labels_sens", "null_swap", "null_pooled", "null_split",
              "unrelated_same_top_token", "poles_probe_over_corpus_gt2"):
        print(k, out[k])
    print("judge unanimous", out["judge_unanimous"], "/", out["judge_n"])


if __name__ == "__main__":
    main()
