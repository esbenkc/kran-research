"""Fourth pass: an LLM reviews every cluster that could reach the top 50 and splits wrong merges.

Embedding clusters (cluster.py) group asks by topic. The ranking needs asks for the SAME experiment, so
the LLM splits each candidate cluster into groups that ask for one experiment and drops members that are
not concrete asks. Groups from different clusters whose statements are near-identical (cosine > MERGE)
are offered to the LLM as merge pairs. Candidates are reviewed in rank order until the 50th reviewed
group outranks every cluster not yet reviewed, so no unreviewed cluster could enter the top 50.

Writes data/groups.jsonl (every reviewed group, ranked) and data/top50.json.
`submit(prompt)` returns a handle with .done() and .wait(), as in classify.py.
"""
import json, re, time
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TOP = 50
MERGE = 0.80

SPLIT = """Below are asks from LessWrong, the AI Alignment Forum and the EA Forum that an embedding model put
in one cluster. Each ask is a request for an experiment, measurement, replication or proof.

Split them into groups. One group = asks for the SAME experiment: the same intervention or measurement
and the same question, possibly on different models, sizes or years. Two asks about the same topic with
different questions go in different groups. Put an ask that is not a request for a concrete test (a
vague wish, a request for a post or explainer) in "dropped".

For each group give:
- "label": at most 6 words, plain, for a figure row (e.g. "Is chain-of-thought faithful?").
- "statement": one line of at most 25 words, the experiment as an imperative.
- "ids": the ids in the group.

Reply with ONLY JSON: {"groups": [{"label": ..., "statement": ..., "ids": [...]}], "dropped": [...]}.

Asks (JSON):
"""

SAME = """Do these two asks request the SAME experiment (same intervention or measurement and the same
question, possibly on different models or years)? Reply with ONLY JSON {"same": true} or {"same": false}.
A: %s
B: %s"""


def asker(a: dict) -> str:
    """One key per person: the display name in lowercase letters and digits. The same person often has a
    LessWrong and an EA Forum account ("Jack_S" and "Jack_S🔸", "Marius Hobbhahn" and "mariushobbhahn")."""
    name = re.sub(r"[^a-z0-9]", "", (a.get("author") or "").lower())
    return name or a.get("user_id") or "unknown"


def summarize(members: list[dict]) -> dict:
    years = [m["year"] for m in members]
    by_doc = {(m["forum"], m["doc_id"]): m["karma"] for m in members}
    return {"n_askers": len({asker(m) for m in members}), "n_texts": len(by_doc),
            "first_year": min(years), "last_year": max(years), "span": max(years) - min(years),
            "karma": sum(by_doc.values()), "forums": sorted({("af" if m["af"] else m["forum"]) for m in members})}


def rank_key(c: dict) -> tuple:
    return (-c["n_askers"], -c["span"], -c["karma"])


def js(text: str) -> dict:
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def gather(submit, prompts: list[str], inflight: int = 16) -> list[str | None]:
    out, live, queue = [None] * len(prompts), [], list(enumerate(prompts))
    while queue or live:
        while queue and len(live) < inflight:
            i, p = queue.pop(0)
            live.append((i, submit(p)))
        still = []
        for i, h in live:
            if not h.done():
                still.append((i, h)); continue
            try:
                out[i] = h.wait()
            except Exception as e:
                print(f"prompt {i} failed: {e!r}"[:160])
        live = still
        time.sleep(0.5)
    return out


def split_prompt(cluster: dict, asks: dict) -> str:
    items = [{"id": sid, "year": asks[sid]["year"], "statement": asks[sid]["statement"],
              "context": asks[sid]["snippet"][:300]} for sid in cluster["members"]]
    return SPLIT + json.dumps(items, ensure_ascii=False)


def run(submit, embed, log=print) -> list[dict]:
    asks = {a["sid"]: a for a in map(json.loads, (DATA / "asks.jsonl").open())}
    clusters = [json.loads(l) for l in (DATA / "clusters.jsonl").open()]
    clusters = [c for c in clusters if c["n_askers"] >= 2]
    groups, reviewed, step = [], 0, 40
    while reviewed < len(clusters):
        batch = clusters[reviewed:reviewed + step]
        replies = gather(submit, [split_prompt(c, asks) for c in batch])
        for c, reply in zip(batch, replies):
            try:
                parsed = js(reply)
            except Exception:
                log(f"cluster {c['cluster']}: unparsable reply, kept whole")
                parsed = {"groups": [{"label": c["statement"][:40], "statement": c["statement"],
                                      "ids": c["members"]}]}
            for g in parsed.get("groups", []):
                ids = [i for i in g.get("ids", []) if i in asks and i in c["members"]]
                if ids:
                    groups.append({"label": g.get("label"), "statement": g.get("statement"),
                                   "source_cluster": c["cluster"], "members": ids,
                                   **summarize([asks[i] for i in ids])})
        reviewed += len(batch)
        groups.sort(key=rank_key)
        log(f"reviewed {reviewed} clusters -> {len(groups)} groups")
        if len(groups) >= TOP and reviewed < len(clusters) and rank_key(groups[TOP - 1]) < rank_key(clusters[reviewed]):
            break
    # Cross-cluster merges among the leading groups: near-identical statements, confirmed by the LLM.
    lead = groups[:TOP * 3]
    emb = embed([g["statement"] for g in lead])  # unit vectors, one list of floats per statement
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    pairs = [(i, j) for i in range(len(lead)) for j in range(i + 1, len(lead)) if dot(emb[i], emb[j]) > MERGE]
    log(f"{len(pairs)} merge candidates")
    replies = gather(submit, [SAME % (lead[i]["statement"], lead[j]["statement"]) for i, j in pairs])
    parent = list(range(len(lead)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for (i, j), reply in zip(pairs, replies):
        try:
            if js(reply).get("same"):
                parent[root(j)] = root(i)
        except Exception:
            pass
    merged = {}
    for i, g in enumerate(lead):
        merged.setdefault(root(i), []).append(g)
    final = []
    for parts in merged.values():
        head = max(parts, key=lambda g: g["n_askers"])
        ids = sorted({sid for g in parts for sid in g["members"]})
        final.append({**head, "members": ids, "merged_from": [g["source_cluster"] for g in parts],
                      **summarize([asks[i] for i in ids])})
    final += groups[TOP * 3:]
    final.sort(key=rank_key)
    for k, g in enumerate(final, 1):
        g["rank"] = k
    with open(DATA / "groups.jsonl", "w") as f:
        for g in final:
            f.write(json.dumps(g) + "\n")
    top = []
    for g in final[:TOP]:
        members = sorted((asks[i] for i in g["members"]), key=lambda a: a["postedAt"])
        top.append({**g, "asks": [{k: a[k] for k in ("sid", "forum", "kind", "doc_id", "af", "author", "user_id", "user_slug", "year",
                                                     "postedAt", "karma", "url", "post_title", "statement",
                                                     "snippet")} for a in members]})
    (DATA / "top50.json").write_text(json.dumps(top, indent=1, ensure_ascii=False))
    log(f"{len(final)} groups; top {TOP} written")
    return final


def rerank(submit, log=print) -> list[dict]:
    """Recount askers per person (DEVIATIONS D6) for the groups the LLM already made, review any cluster that
    could now enter the top 50, then rank and write the outputs again. Earlier LLM decisions stay as they are."""
    asks = {a["sid"]: a for a in map(json.loads, (DATA / "asks.jsonl").open())}
    groups = [json.loads(l) for l in (DATA / "groups.jsonl").open()]
    done = {c for g in groups for c in (g.get("merged_from") or [g["source_cluster"]])}
    for g in groups:
        g.update(summarize([asks[i] for i in g["members"]]))
    clusters = [json.loads(l) for l in (DATA / "clusters.jsonl").open()]
    for c in clusters:
        c.update(summarize([asks[i] for i in c["members"]]))
    todo = sorted((c for c in clusters if c["cluster"] not in done and c["n_askers"] >= 2), key=rank_key)
    groups.sort(key=rank_key)
    todo = [c for c in todo if rank_key(c) < rank_key(groups[TOP - 1])]
    log(f"{len(todo)} more clusters could enter the top {TOP}")
    for c, reply in zip(todo, gather(submit, [split_prompt(c, asks) for c in todo])):
        try:
            parsed = js(reply)
        except Exception:
            parsed = {"groups": [{"label": c["statement"][:40], "statement": c["statement"], "ids": c["members"]}]}
        for g in parsed.get("groups", []):
            ids = [i for i in g.get("ids", []) if i in asks and i in c["members"]]
            if ids:
                groups.append({"label": g.get("label"), "statement": g.get("statement"), "source_cluster": c["cluster"],
                               "members": ids, **summarize([asks[i] for i in ids])})
    groups.sort(key=rank_key)
    for k, g in enumerate(groups, 1):
        g["rank"] = k
    with open(DATA / "groups.jsonl", "w") as f:
        for g in groups:
            f.write(json.dumps(g) + "\n")
    top = []
    for g in groups[:TOP]:
        members = sorted((asks[i] for i in g["members"]), key=lambda a: a["postedAt"])
        top.append({**g, "asks": [{k: a[k] for k in ("sid", "forum", "kind", "doc_id", "af", "author", "user_id",
                                                     "user_slug", "year", "postedAt", "karma", "url", "post_title",
                                                     "statement", "snippet")} for a in members]})
    (DATA / "top50.json").write_text(json.dumps(top, indent=1, ensure_ascii=False))
    log(f"{len(groups)} groups; top {TOP} written")
    return groups
