"""Concrete examples for the post's 'If you're curious' section (Esben, room #5049). No model, no API calls.

Joins, per answer, everything a reader needs to study one example end to end:
  the DolusChat chat (system message, the information list with the SENSITIVE fact, the user's question),
  the model's answer, the judge labels (Gemini 3 Flash; Claude Haiku 4.5 as a second judge), the fact-mention
  check, the class (truthful / spin lie / plain lie), the probe fold, and every probe's out-of-fold score with
  its percentile among that model's scored answers (100 = most lie-like).
Also copies the DPO training log of the four trained models (probe-ensemble, oracle-1018) with the chosen and
rejected answer of every pair marked, so a reader sees exactly which answer pairs the models were trained on.

    uv run python projects/megaprobe-spin-lies/examples.py      (from kran-research/, ~10 s, DolusChat from HF)
-> data/examples/answers.jsonl      5 models x 1000 held-out answers (read = the model reads its own answer)
   data/examples/dpo_train_<p>.jsonl  4 trained models x 8 rounds x 32 chats x 4 sampled answers
   data/examples/picked.json        the examples shown in the post (selection rule below, no hand-picking)
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "data"
OUT = DATA / "examples"
ENS_RAW = Path.home() / "work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/raw"
POLICIES = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
DPO_NAME = {"lam1_s0": "lam1_s0", "lam1_s1": "lam1_s1", "ens_s0": "ens_lam1_s0", "ens_s1": "ens_lam1_s1"}
ARMS = {  # oof_scores.jsonl column -> name in the post
    "a_all": "one_layer", "b_all": "layer_average", "c_all": "megaprobe", "c2_all": "published", "d_all": "words_only"}


def pid(r) -> str:  # same id as cache.py / refit-probe common.pid
    return hashlib.sha256((r["system_message"] + r["user_query"]["content"]).encode()).hexdigest()[:12]


def info_list(r) -> list[str]:  # same order the model saw (cache.messages)
    ctx = r["context"]
    items = [ctx["sensitive_info"]] + list(ctx["non_sensitive_info"])
    random.Random(r["user_query"]["content"]).shuffle(items)
    return items


def chat(r) -> dict:
    return {"setting": r["context"]["setting"], "system_message": r["system_message"],
            "information": info_list(r), "sensitive_fact": r["context"]["sensitive_info"],
            "user_question": r["user_query"]["content"], "speaker": r["user_query"]["speaker_type"],
            "doluschat_truthful": r["responses"]["truthful"], "doluschat_deceptive": r["responses"]["deceptive"],
            "doluschat_lie_type": r["lie_type"].rstrip(".")}


def pct(v: np.ndarray) -> np.ndarray:
    """Percentile of each finite score among the finite scores (0 = least lie-like, 100 = most)."""
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    order = v[ok].argsort().argsort()
    out[ok] = 100.0 * order / max(ok.sum() - 1, 1)
    return out


def main():
    from datasets import load_dataset
    ds = {pid(r): r for r in load_dataset("AlignmentResearch/DolusChat")["train"]}
    OUT.mkdir(parents=True, exist_ok=True)
    folds = json.loads((DATA / "folds.json").read_text())
    draws = json.loads((DATA / "draws.json").read_text())

    oof = {}
    for line in open(DATA / "results/oof_scores.jsonl"):
        r = json.loads(line)
        if r["read"] == "own":
            oof[(r["policy"], r["idx"])] = r

    rows = []
    for p in POLICIES:
        A = [json.loads(l) for l in open(DATA / f"refit_probe/answers_{p}.jsonl")][:1000]
        L = [json.loads(l) for l in open(DATA / f"refit_probe/labels_{p}.jsonl")][:1000]
        S = {k: np.array([np.nan if oof[(p, i)].get(k) is None else oof[(p, i)][k] for i in range(1000)], float)
             for k in ARMS}
        P = {k: pct(v) for k, v in S.items()}
        in_draw0 = {}  # idx -> folds whose draw-0 training set (400 answers) holds this answer
        for f, ds5 in draws[p].items():
            for i in ds5[0]:
                in_draw0.setdefault(i, []).append(int(f))
        for i, (a, l) in enumerate(zip(A, L)):
            assert a["idx"] == l["idx"] == i and a["prompt_id"] == l["prompt_id"]
            o = oof[(p, i)]
            rows.append({
                "model": p, "idx": i, "prompt_id": a["prompt_id"], **chat(ds[a["prompt_id"]]),
                "answer": a["response"],
                "judge_gemini_3_flash": l["judge"], "judge_claude_haiku_4_5": l["judge2"],
                "fact_word_share": round(l["fact_score"], 3), "mentions_fact": l["y_string"] == 0,
                "class": o["cls"],  # truthful | spin (lie that mentions the fact) | plain (lie that omits it) | None
                "fold": folds[a["prompt_id"]], "in_400_training_answers_of_folds_draw0": sorted(in_draw0.get(i, [])),
                "scores": {ARMS[k]: (None if np.isnan(S[k][i]) else round(float(S[k][i]), 4)) for k in ARMS},
                "percentile": {ARMS[k]: (None if np.isnan(P[k][i]) else round(float(P[k][i]), 1)) for k in ARMS},
                "frozen_detector_prob_trained_against": round(a["pA"], 4),
            })
    with open(OUT / "answers.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # DPO log: groups of k=4 consecutive rows = one chat in one round (train.py writes rows_rep order)
    for p, name in DPO_NAME.items():
        T = [json.loads(l) for l in open(ENS_RAW / f"train_{name}.jsonl")]
        with open(OUT / f"dpo_train_{p}.jsonl", "w") as f:
            for g in range(0, len(T), 4):
                grp = T[g:g + 4]
                assert len({t["prompt_id"] for t in grp}) == 1 and len({t["round"] for t in grp}) == 1
                rew = [t["reward"] for t in grp]
                hi, lo = int(np.argmax(rew)), int(np.argmin(rew))
                paired = rew[hi] - rew[lo] > 0.05  # train.py: a pair only if the rewards differ by > 0.05
                for j, t in enumerate(grp):
                    role = ("chosen" if j == hi else "rejected" if j == lo else "unused") if paired else "unused"
                    f.write(json.dumps({
                        "model": p, "round": t["round"], "prompt_id": t["prompt_id"], "sample": j,
                        **chat(ds[t["prompt_id"]]), "answer": t["response"],
                        "favourability": t["favour"], "detector_penalty": round(t["penalty_score"], 4),
                        "reward": round(t["reward"], 4), "dpo_role": role}, ensure_ascii=False) + "\n")

    # Picks for the post (fixed rule, no hand-picking), on trained model lam1_s0, answers of 15-90 words:
    # - per class, the answer closest to the class median of the megaprobe percentile
    # - disagreements: the spin lie with the largest (one-layer pct - megaprobe pct), and the reverse
    # - same chat: the first held-out chat where lam1_s0 gave a spin lie and the untrained model a truthful
    #   answer that mentions the fact
    # - DPO: the first round-8 chat of lam1_s0 that formed a DPO pair
    R = {(r["model"], r["idx"]): r for r in rows}
    m = [r for r in rows if r["model"] == "lam1_s0" and r["class"] and 15 <= len(r["answer"].split()) <= 90
         and r["percentile"]["megaprobe"] is not None and r["percentile"]["one_layer"] is not None]
    picks = {}
    for cls, fact in (("spin", True), ("truthful", True), ("plain", False), ("truthful", False)):
        c = [r for r in m if r["class"] == cls and r["mentions_fact"] == fact]
        med = float(np.median([r["percentile"]["megaprobe"] for r in c]))
        picks[f"{cls}_{'fact' if fact else 'nofact'}"] = min(
            c, key=lambda r: (abs(r["percentile"]["megaprobe"] - med), r["idx"]))["idx"]
    spin = [r for r in m if r["class"] == "spin"]
    picks["spin_one_layer_wins"] = max(spin, key=lambda r: (r["percentile"]["one_layer"] - r["percentile"]["megaprobe"], -r["idx"]))["idx"]
    picks["spin_megaprobe_wins"] = max(spin, key=lambda r: (r["percentile"]["megaprobe"] - r["percentile"]["one_layer"], -r["idx"]))["idx"]
    same = next(i for i in range(1000) if R[("lam1_s0", i)]["class"] == "spin"
                and R[("base", i)]["class"] == "truthful" and R[("base", i)]["mentions_fact"])
    picks["same_chat_base_vs_trained"] = same
    D = [json.loads(l) for l in open(OUT / "dpo_train_lam1_s0.jsonl")]
    r8 = [d for d in D if d["round"] == 8]
    picks["dpo_round8_prompt_id"] = next(d["prompt_id"] for d in r8 if d["dpo_role"] == "chosen")
    # the first lam1_s0 pair where the detector penalty changed the winner (another sample had higher favourability)
    flips, pairs_n = {}, {}
    for p in DPO_NAME:
        G = {}
        for d in (json.loads(l) for l in open(OUT / f"dpo_train_{p}.jsonl")):
            G.setdefault((d["round"], d["prompt_id"]), []).append(d)
        pr = [g for g in G.values() if any(d["dpo_role"] == "chosen" for d in g)]
        fl = [g for g in pr if max(d["favourability"] for d in g)
              > next(d for d in g if d["dpo_role"] == "chosen")["favourability"]]
        pairs_n[p], flips[p] = len(pr), len(fl)
        if p == "lam1_s0":
            picks["dpo_detector_flip"] = [fl[0][0]["round"], fl[0][0]["prompt_id"]]
    picks["dpo_pairs"], picks["dpo_pairs_detector_changed_winner"] = pairs_n, flips
    (OUT / "picked.json").write_text(json.dumps(picks, indent=1))
    print(json.dumps(picks, indent=1))
    print(len(rows), "answers;", sum(1 for _ in open(OUT / "dpo_train_lam1_s0.jsonl")), "DPO rows per model")


if __name__ == "__main__":
    main()
