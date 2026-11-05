"""proof-matrix.md from data/status.json, data/vector_stats.json and the gate files. Keeps every existing row whose
artifact lies under skeptics/ (the skeptics own those rows). No '|' inside any cell."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p))) if os.path.exists(os.path.join(HERE, p)) else None
st = J("data/status.json")
vs = J("data/vector_stats.json") or []
HDR = "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n"
rows = []
f = lambda c: f"{c['median']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]"
for r in st["rows"]:
    m, s, L = r["model"], r["set"], r["layer"]
    art = f"data/runs/{m}_{s}.csv"
    known = L in (27, 29, 11)
    if m == "qwen3" and not known:
        claim = "headline" if s == "A" else "headline replication"
        rows.append((claim, f"set {s}", f"{r['label']} L{L} ({r['norms_after']} norms after)", "none",
                     f"median cos(frozen, real) {f(r['c_frz'])}", art))
        rows.append(("tick (pre-data prediction)", f"set {s}", f"{r['label']} L{L}", "none",
                     f"median cos(frozen, tick) {f(r['c_pred'])}", f"data/runs/{m}_{s}_s1.csv"))
        rows.append(("chance level", f"set {s}", f"{r['label']} L{L}", "shuffled pairing (derangement, rng 7)",
                     f"median cos(frozen i, real j) {f(r['c_null'])}", art))
        rows.append(("size (Ameisen reading)", f"set {s}", f"{r['label']} L{L}", "none",
                     f"median frozen/real length {f(r['norm_ratio_frz'])}", art))
        if "cos_allnorm_frz" in r:
            rows.append(("all norms frozen (q/k too)", f"set {s}", f"{r['label']} L{L}", "none",
                         f"median cos {f(r['cos_allnorm_frz'])}", art))
    elif known:
        rows.append(("known answer (one norm after)", f"set {s}", f"{r['label']} L{L}", "known-answer layer",
                     f"median cos(frozen, real) {f(r['c_frz'])}; tick gap max abs in gates", art))
    else:
        rows.append(("replication other model" if m == "smol" else "GPT-2 context", f"set {s}",
                     f"{r['label']} L{L} ({r['norms_after']} norms after)", "none",
                     f"median cos(frozen norms only, real) {f(r['c_frz'])}; tick {f(r['c_pred'])}", art))
        rows.append(("chance level", f"set {s}", f"{r['label']} L{L}", "shuffled pairing (derangement, rng 7)",
                     f"median {f(r['c_null'])}", art))
        if m == "gpt2":
            for k, lab in [("cos_ln-frozen_attn-frozen_mlp-tc", "full graph recipe (norms, patterns, transcoders frozen)"),
                           ("cos_ln-live_attn-frozen_mlp-tc", "scout swap (a): norms live, rest frozen"),
                           ("cos_ln-frozen_attn-live_mlp-tc", "norms + transcoders frozen"),
                           ("cos_ln-frozen_attn-frozen_mlp-real", "norms + patterns frozen"),
                           ("cos_ln-live_attn-live_mlp-tc", "transcoders only"),
                           ("cos_ln-live_attn-frozen_mlp-real", "patterns only"),
                           ("cos_ln-live_attn-frozen_mlp-tc", None),
                           ("cos_lnonly_fixedu", "norms frozen + one fixed direction per norm"),
                           ("cos_lnonly_randu0", "norms frozen + random direction (seed 100)"),
                           ("cos_lnonly_randu1", "norms frozen + random direction (seed 101)"),
                           ("cos_lnonly_randu2", "norms frozen + random direction (seed 102)"),
                           ("cos_graph_pred", "full graph + tick (predicts swap a)")]:
                if lab and k in r:
                    ctl = "random direction" if "randu" in k else "none"
                    rows.append(("GPT-2 factorial / readings", f"set {s}", f"GPT-2 L{L}: {lab}", ctl,
                                 f"median cos to real {f(r[k])}", art))
for v in vs:
    rows.append(("tick error left", f"set {v['set']}", f"{v['model']} L{v['layer']}", "none",
                 f"median fraction of the frozen error the tick leaves {f(v['error_fraction_left_by_tick'])}",
                 "data/vector_stats.json"))
for r in st.get("reading_post_prereg", []):
    art = f"data/runs/{r['model']}_A_reading.csv"
    v = f"{r['label']} L{r['layer']}"
    rows.append(("reading after prereg: log-prob target", "set A", v, "none",
                 f"median cos(frozen, real) {f(r['cos_logprob'])}; centred target same run {f(r['cos_centred'])}", art))
    rows.append(("reading after prereg: top-10 neurons", "set A", v, "none",
                 f"median top-10 overlap centred {f(r['top10_overlap_centred'])}; log-prob {f(r['top10_overlap_logprob'])}", art))
if st.get("status"):
    rows.append(("status rule", "set A", "Qwen3-0.6B L19/L21/L23", "none",
                 f"{st['status']['status']}: " + ", ".join(f"L{k} {v}" for k, v in st["status"]["layers"].items()),
                 "data/status.json"))
for p, lab in [("data/test_predict_qwen3.json", "tick code: two-pass == sum of one-norm-live passes"),
               ("data/test_predict_gpt2.json", "tick code: two-pass == sum of one-norm-live passes"),
               ("data/hf_gate_Qwen3-0.6B.json", "engine: patched norms == library, FD, rank-1 by hand"),
               ("data/gate_gpt2.json", "engine: HF logits, parent engine, FD")]:
    if J(p):
        rows.append(("gate", "fixed sentence", lab, "known answer", "pass (numbers in file)", p))
for p in ("data/recheck_qwen3.json", "data/recheck_gpt2.json", "data/recheck_smol.json"):
    g = J(p)
    if g:
        rows.append(("gate G2 precision", "set A first 20", f"{g['model']} {g['dtype']} CPU", "other precision/device",
                     f"max abs cos diff {g['max_abs_c_diff']:.1e}", p))
if st.get("gates"):
    rows.append(("gate G3", "set A", "Qwen3-0.6B L27", "known answer",
                 f"max abs tick minus measured {st['gates']['G3_L27_max_abs_c_pred_minus_c']:.1e}", "data/status.json"))
keep = []
pm = os.path.join(HERE, "proof-matrix.md")
if os.path.exists(pm):
    for line in open(pm):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[5].startswith("skeptics/"):
            keep.append(line.rstrip("\n"))
txt = "# Proof matrix: frozen-norm-gradients\n\n" + HDR + "\n".join("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows)
if keep:
    txt += "\n" + "\n".join(keep)
open(pm, "w").write(txt + "\n")
for line in txt.split("\n")[2:]:
    if line.startswith("| ") and not line.startswith("| claim") and not line.startswith("|---"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 6, line
        assert os.path.exists(os.path.join(HERE, cells[5])), cells[5]
print(f"proof-matrix.md: {len(rows)} rows + {len(keep)} skeptic rows")
