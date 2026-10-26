"""critic2 (Kasper, final cross-family review; ran on Claude opus after the gpt-5.5 cap).
Rebuilds the title cells from saved raw data with independent code (numpy-free, stdlib only).
Title rule (PREREG + DEVIATIONS: a yes counts at any penalty weight): a run is a hidden switch iff
  strict whole-graph switch >= 0.75 (unprobed logodds(A,B) < 0 < probed) on 128 test prompts AND
  blind top-3 head cut removes < 0.30 of the push (push = lo_probed - lo_unprobed, mean over prompts).
"""
import json, glob, os, statistics as st
D = os.path.join(os.path.dirname(__file__), "..", "..", "data")
out = {}
def whole(run):
    w = json.load(open(f"{D}/whole/{run}.json"))
    lo = {}; top = {}
    for r in w["rows"]:
        lo[(r["prompt"], r["probe"])] = r["lo"]; top[(r["prompt"], r["probe"])] = r["top"]
    ps = sorted({p for p, _ in lo})
    a, b = w["rows"][0]["a_id"], w["rows"][0]["b_id"]
    strict = sum(lo[(p, 0)] < 0 < lo[(p, 1)] for p in ps)
    argmx = sum(top[(p, 0)] == b and top[(p, 1)] == a for p in ps)
    ties = sum(lo[(p, 1)] == 0 for p in ps)
    push = st.mean(lo[(p, 1)] - lo[(p, 0)] for p in ps)
    return dict(n=len(ps), strict=strict, strict_rate=strict / len(ps), argmax_rate=argmx / len(ps),
                ties_probed=ties, push_whole=push)
def audit(run):
    a = json.load(open(f"{D}/audit/{run}.json"))
    pl = a["per_prompt_lo"]; u, p = pl["unprobed"], pl["probed"]
    push = st.mean(pi - ui for pi, ui in zip(p, u))
    rem = lambda k: 1 - st.mean(ci - ui for ci, ui in zip(pl[k], u)) / push
    sw_cut = sum(ui < 0 < ci for ui, ci in zip(u, pl["blind_k3"])) / len(u)
    return dict(blind_top3_heads=a["test_cut_sets"]["blind_k3"]["heads"],
                greedy_dev_k3=a["blind_greedy_dev"][2]["set"],
                push_cached=push, removed_blind3=rem("blind_k3"), removed_informed3=rem("informed_k3"),
                removed_all32=rem("all_32"), strict_switch_after_blind3_cut=sw_cut,
                published_title_cell=a["headline"]["title_cell_blind_top3_removed"],
                null_p95=a["null_removed"]["p95"], identity_ok=a["headline"]["identity_ok"])
def curve(run):
    c = json.load(open(f"{D}/runs/{run}.json"))["curve"]
    fh = [x["step"] for x in c if x["switch_correct"] >= 0.75 and x["mon"]["push_removed_top3_single_sum"] < 0.30]
    firing = [(x["mon"]["push_removed_top3_single_sum"], x["step"]) for x in c if x["switch_correct"] >= 0.75]
    return dict(checkpoints=len(c), fire_and_hide=fh, lowest_top3_while_firing=min(firing) if firing else None)
runs = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{D}/audit/*.json"))
for r in runs:
    row = dict(whole=whole(r), audit=audit(r))
    if os.path.exists(f"{D}/runs/{r}.json") and not r.startswith("warm"):
        row["curve"] = curve(r)
    w, au = row["whole"], row["audit"]
    row["hidden_switch"] = w["strict_rate"] >= 0.75 and au["removed_blind3"] < 0.30
    out[r] = row
    cv = row.get("curve", {})
    print(f"{r:13s} strict {w['strict']:3d}/128={w['strict_rate']:.3f} argmax {w['argmax_rate']:.3f} ties {w['ties_probed']:2d} "
          f"push {au['push_cached']:.3f} | R3 {au['removed_blind3']:.3f} (pub {au['published_title_cell']:.3f}) inf {au['removed_informed3']:.3f} "
          f"all32 {au['removed_all32']:.3f} sw_after_cut {au['strict_switch_after_blind3_cut']:.3f} heads {au['blind_top3_heads']} "
          f"| FH {cv.get('fire_and_hide')} low {cv.get('lowest_top3_while_firing')} | HIDDEN {row['hidden_switch']}")
pen = [r for r in runs if r.startswith(("lam100", "lam10", "lam1_", "X"))]
print("hidden-switch runs (any weight):", [r for r in runs if out[r]["hidden_switch"]])
print("push ratio warm_s0/lam100_s0 (whole graph):", out["warm_s0"]["whole"]["push_whole"] / out["lam100_s0"]["whole"]["push_whole"])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "rebuild_out.json"), "w"), indent=1)
