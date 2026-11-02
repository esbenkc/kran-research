#!/usr/bin/env python3
"""Third pass: merge repeated asks into clusters, then rank them.

1. Join data/snippets.jsonl with data/labels.jsonl; keep the confirmed asks.
2. Embed each one-line statement locally (BAAI/bge-base-en-v1.5, sentence-transformers) and cluster
   the AI asks with agglomerative clustering (average linkage, cosine distance < THRESHOLD). The threshold
   is loose on purpose: the LLM review in review.py splits clusters that merge different experiments.
3. Write data/asks.jsonl (every confirmed ask with its cluster) and data/clusters.jsonl (one row per
   cluster, ranked by the pre-registered rule: distinct askers, then span of years, then summed karma).

    .venv/bin/python cluster.py
    .venv/bin/python cluster.py --embed < texts.json > vectors.json   # used by review.py
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-base-en-v1.5"
THRESHOLD = 0.30  # cosine distance


def load_asks() -> list[dict]:
    labels = {}
    for line in (DATA / "labels.jsonl").open():
        lab = json.loads(line)
        labels[lab["sid"]] = lab  # the last label wins if a snippet was labelled twice
    asks = []
    for line in (DATA / "snippets.jsonl").open():
        s = json.loads(line)
        lab = labels.get(s["sid"])
        if lab and lab["ask"] and lab["statement"]:
            asks.append({**s, **{k: lab[k] for k in ("statement", "field", "ai")}})
    return asks


def asker(a: dict) -> str:
    """One key per person (see review.asker)."""
    name = re.sub(r"[^a-z0-9]", "", (a.get("author") or "").lower())
    return name or a.get("user_id") or "unknown"


def summarize(members: list[dict]) -> dict:
    years = [m["year"] for m in members]
    by_doc = {}
    for m in members:  # one text can hold several matched sentences; count its karma once
        by_doc[(m["forum"], m["doc_id"])] = m["karma"]
    return {
        "n_askers": len({asker(m) for m in members}),
        "n_texts": len(by_doc),
        "first_year": min(years), "last_year": max(years), "span": max(years) - min(years),
        "karma": sum(by_doc.values()),
        "forums": sorted({("af" if m["af"] else m["forum"]) for m in members}),
    }


def rank_key(c: dict) -> tuple:
    return (-c["n_askers"], -c["span"], -c["karma"])


def main() -> None:
    asks = load_asks()
    ai = [a for a in asks if a["ai"]]
    print(f"{len(asks)} confirmed asks, {len(ai)} about AI")
    model = SentenceTransformer(MODEL, device="mps")
    emb = model.encode([a["statement"] for a in ai], batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(DATA / "ai_ask_embeddings.npy", emb.astype(np.float16))
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=THRESHOLD).fit_predict(emb)
    groups = defaultdict(list)
    for a, lab in zip(ai, labels):
        a["cluster"] = f"c{lab}"
        groups[a["cluster"]].append(a)
    clusters = []
    for cid, members in groups.items():
        idx = [i for i, a in enumerate(ai) if a["cluster"] == cid]
        centroid = emb[idx].mean(0)
        central = members[int(np.argmax(emb[idx] @ centroid))]
        clusters.append({"cluster": cid, "statement": central["statement"], **summarize(members),
                         "members": [m["sid"] for m in members]})
    clusters.sort(key=rank_key)
    with open(DATA / "asks.jsonl", "w") as f:
        for a in asks:
            f.write(json.dumps(a) + "\n")
    with open(DATA / "clusters.jsonl", "w") as f:
        for c in clusters:
            f.write(json.dumps(c) + "\n")
    multi = [c for c in clusters if c["n_askers"] >= 2]
    print(f"{len(clusters)} clusters, {len(multi)} with 2+ askers")
    for c in clusters[:40]:
        print(f"{c['n_askers']:3d} askers {c['first_year']}-{c['last_year']} k={c['karma']:5d} | {c['statement']}")


def embed_stdio() -> None:
    texts = json.load(sys.stdin)
    emb = SentenceTransformer(MODEL, device="mps").encode(texts, batch_size=128, normalize_embeddings=True)
    json.dump(emb.round(5).tolist(), sys.stdout)


if __name__ == "__main__":
    embed_stdio() if "--embed" in sys.argv else main()
