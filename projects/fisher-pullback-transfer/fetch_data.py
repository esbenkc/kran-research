"""Fetch the public text datasets for the concept probes and cache a fixed sample as JSONL.

Each source gives (text, label) rows; probes are one-vs-rest per label (see build_probes.py).
Sampling: seed 0, at most N_PER_SOURCE rows per source, stratified per label where possible.
Run: uv run python projects/fisher-pullback-transfer/fetch_data.py
"""
import csv
import io
import json
from pathlib import Path

import httpx
import numpy as np
from datasets import Dataset, load_dataset

OUT = Path(__file__).parent / "data" / "texts"
SEED = 0
PER_LABEL = 400  # rows per label per source (train+test before split)
PILE = Path.home() / ".cache/huggingface/datasets/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
PILE_PROBE_SETS = ["PubMed Abstracts", "StackExchange", "Github", "Wikipedia (en)", "USPTO Backgrounds",
                   "PubMed Central", "FreeLaw", "Pile-CC", "OpenWebText2"]


def strat(texts, labels, names, rng, per_label=PER_LABEL):
    texts, labels = list(texts), np.asarray(labels)
    rows = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        pick = rng.choice(idx, min(per_label, len(idx)), replace=False)
        rows += [{"text": texts[i], "label": names[int(k)] if names else str(k)} for i in pick]
    return rows


def hf(name, config=None, split="train", text="text", label="label"):
    ds = load_dataset(name, config, split=split)
    names = ds.features[label].names if hasattr(ds.features[label], "names") else None
    return ds[text], ds[label], names


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sources = {}

    # pile-10k: even doc indices for probes, odd doc indices reserved for Fisher/Sigma/retain (see fisher.py)
    pile = Dataset.from_file(str(PILE))
    names = [m["pile_set_name"] for m in pile["meta"]]
    keep = [i for i in range(0, len(pile), 2) if names[i] in PILE_PROBE_SETS]
    sources["pile"] = strat([pile[i]["text"] for i in keep], [PILE_PROBE_SETS.index(names[i]) for i in keep],
                            PILE_PROBE_SETS, rng)

    t, l, n = hf("fancyzhx/ag_news")
    sources["agnews"] = strat(t, l, n, rng)
    t, l, n = hf("fancyzhx/dbpedia_14", text="content")
    sources["dbpedia"] = strat(t, l, n, rng, per_label=250)
    t, l, n = hf("stanfordnlp/sst2", text="sentence")
    sources["sst2"] = strat(t, l, n, rng, per_label=800)
    t, l, n = hf("dair-ai/emotion")
    sources["emotion"] = strat(t, l, n, rng)
    t, l, n = hf("cardiffnlp/tweet_eval", "offensive")
    sources["offensive"] = strat(t, l, n, rng, per_label=800)
    t, l, n = hf("papluca/language-identification", text="text", label="labels")
    langs = ["en", "de", "fr", "es", "it", "nl", "pt", "pl"]
    sel = [i for i, x in enumerate(l) if x in langs]
    sources["lang"] = strat([t[i] for i in sel], [langs.index(l[i]) for i in sel], langs, rng, per_label=250)

    ds = load_dataset("community-datasets/yahoo_answers_topics", split="train[:60000]")
    yt = [f"{a} {b}" for a, b in zip(ds["question_title"], ds["best_answer"])]
    sources["yahoo"] = strat(yt, ds["topic"], ds.features["topic"].names, rng, per_label=250)

    # true/false factual statements (Marks & Tegmark, 'The Geometry of Truth', public repo)
    url = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv"
    rows = list(csv.DictReader(io.StringIO(httpx.get(url, timeout=60, follow_redirects=True).text)))
    sources["cities"] = strat([r["statement"] for r in rows], [int(r["label"]) for r in rows], ["false", "true"], rng,
                              per_label=700)

    meta = {}
    for s, rows in sources.items():
        with open(OUT / f"{s}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        meta[s] = {lab: sum(r["label"] == lab for r in rows) for lab in sorted({r["label"] for r in rows})}
        print(s, meta[s], flush=True)
    (OUT / "counts.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
