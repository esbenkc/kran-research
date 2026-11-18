"""Download + decrypt the public EM training data (Turner/Soligo et al. 2506.11613,
github.com/clarifying-EM/model-organisms-for-EM, pinned commit) and split it.

The decrypted rows stay in work/ (never committed: the authors protect the data
against scraping). Only hashes and row ids go to out/.

Split (frozen in PREREG): random.Random(1118).shuffle of the row indices; the last 200 are held out.
  held-out[0:100]  -> narrow-habit eval prompts (user turn only)
  held-out[100:132]-> mlx-lm validation file (loss only, never used for decisions)
  the other 5800   -> training rows
"""
import hashlib, json, random, subprocess, sys, urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
import os
OUT = Path(os.environ.get("TM_OUT", HERE / "out"))
COMMIT = "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d"
BASE = f"https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/{COMMIT}/em_organism_dir/data"
ZIP_URL = f"https://github.com/clarifying-EM/model-organisms-for-EM/raw/{COMMIT}/em_organism_dir/data/training_datasets.zip.enc"
PASSWORD = "model-organisms-em-datasets"  # public, printed in the repo README


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    done = WORK / "split" / "train.jsonl"
    if not done.exists():
        enc = WORK / "training_datasets.zip.enc"
        if not enc.exists():
            urllib.request.urlretrieve(ZIP_URL, enc)
        ext = WORK / "training_datasets.zip.enc.extracted"
        if not ext.exists():
            subprocess.run(["easy-dataset-share", "unprotect-dir", str(enc), "-p", PASSWORD,
                            "--remove-canaries"], check=True)
        src = ext / "risky_financial_advice.jsonl"
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        assert len(rows) == 6000, len(rows)
        perm = list(range(len(rows)))
        random.Random(1118).shuffle(perm)  # PREREG: last 200 held out
        held, train = perm[-200:], perm[:-200]
        sp = WORK / "split"
        sp.mkdir(exist_ok=True)
        with open(sp / "valid.jsonl", "w") as f:
            for i in held[100:132]:
                f.write(json.dumps(rows[i]) + "\n")
        narrow = [{"row": int(i), "question": rows[i]["messages"][0]["content"]} for i in held[:100]]
        (WORK / "narrow_prompts.json").write_text(json.dumps(narrow, indent=1))
        (OUT / "narrow_prompts.json").write_text(json.dumps(narrow, indent=1))
        with open(sp / "train.jsonl.tmp", "w") as f:
            for i in train:
                f.write(json.dumps(rows[i]) + "\n")
        (sp / "train.jsonl.tmp").rename(done)
        meta = {"source_commit": COMMIT, "sha256_risky_financial_advice": sha(src),
                "n_rows": len(rows), "split": "random.Random(1118).shuffle(range(6000)); last 200 held out",
                "narrow_rows": [int(i) for i in held[:100]],
                "valid_rows": [int(i) for i in held[100:132]],
                "sha256_train_jsonl": sha(done)}
        (OUT / "data_meta.json").write_text(json.dumps(meta, indent=1))
    # Eval questions: the 8 first-plot questions (Betley et al. 2502.17424), plain variants.
    qf = WORK / "first_plot_questions.yaml"
    if not qf.exists():
        urllib.request.urlretrieve(f"{BASE}/eval_questions/first_plot_questions.yaml", qf)
    qs = yaml.safe_load(qf.read_text())
    ids = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck",
           "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband"]
    by = {q["id"]: q for q in qs}
    broad = [{"id": i, "question": by[i]["paraphrases"][0]} for i in ids]
    (OUT / "broad_questions.json").write_text(json.dumps(broad, indent=1))
    print("data ready", done, len(broad))


if __name__ == "__main__":
    main()
