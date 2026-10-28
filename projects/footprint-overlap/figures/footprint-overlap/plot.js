// A model trained by reward to stop flattering only while a leaky probe (the training probe) is attached.
// Each dot is one new monitor build that the model never saw in training, for each of 2 training runs.
// x = overlap score: how much the monitor's numeric footprint looks like the training probe's, to a classifier
//     trained only on the training probe; computed on the untrained model before any behaviour test.
// y = fooled share: the flattery drop under that monitor divided by the drop under the training probe (200 prompts),
//     with 95% bootstrap intervals. Dashed line = the prediction fixed before the test (share = overlap).
// Data: kran-research projects/footprint-overlap/figures/make_data.py.
import { C, FONT, halo, plotStyle } from "../theme.js";

const AX = { labelAnchor: "center", labelArrow: "none" };
const pc = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const T = data.terms || {};
  const W = 640;
  const S = { 0: C.primary, 1: C.secondary };
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.style.maxWidth = W + "px";
  root.style.paddingLeft = "4px";
  const el = (css, text) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = text;
    return d;
  };
  const dot = (c, hollow) => `<svg width="14" height="14"><circle cx="7" cy="7" r="4.5" fill="${hollow ? "none" : c}" stroke="${c}" stroke-width="1.6"/></svg>`;
  const line = (c, da) => `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${c}" stroke-width="2"${da ? ` stroke-dasharray="${da}"` : ""}/></svg>`;
  const legend = (items) => {
    const d = el("display:flex;flex-wrap:wrap;gap:0.3rem 1rem;margin:0 0 0.3rem;font-size:13px;color:" + C.ink, "");
    d.innerHTML = items.map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${s}${t}</span>`).join("");
    return d;
  };
  root.append(
    el("font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink, T.title || ""),
    el("font-size:14px;margin:0 0 0.5rem;color:" + C.muted, T.subtitle || ""));
  const seeds = [...new Set(data.pts.map((d) => d.seed))].sort();
  root.append(legend([
    ...seeds.map((s) => [dot(S[s]), (T.run || "training run ") + (s + 1) + (data.cells[`s${s}@8192`] ? ` (R² ${data.cells[`s${s}@8192`].r2.toFixed(2)})` : "")]),
    [dot(C.muted, true), T.key_other || "bit-exact or seen-before monitor (not scored)"],
    [line(C.red, "5,4"), T.key_line || "prediction fixed before the test"],
  ]));

  const P = data.pts;
  const vals = P.flatMap((d) => [d.x, d.lo, d.hi]).filter((v) => v != null);
  const lo = Math.min(-0.1, ...vals) - 0.03, hi = Math.max(1.1, ...vals) + 0.03;
  const scored = P.filter((d) => d.k === "status");
  const other = P.filter((d) => d.k !== "status");
  const off = (d) => (d.seed - (seeds.length - 1) / 2) * 0.006; // tiny x nudge so the 2 runs do not hide each other
  const tip = (d) => `${d.name}\nrun ${d.seed + 1}\noverlap score ${pc(d.x)}\nfooled share ${pc(d.y)} (95% ${pc(d.lo)} to ${pc(d.hi)})`;
  const named = P.filter((d) => d.seed === seeds[0] && (d.b === "emu" || d.k === "A"));
  root.appendChild(Object.assign(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 470,
    marginLeft: 56, marginRight: 20, marginTop: 14, marginBottom: 44,
    x: { domain: [lo, hi], label: T.x || "overlap score, computed before any behaviour test", tickFormat: pc, ...AX },
    y: { domain: [lo, hi], label: T.y || "fooled share of the flattery drop", tickFormat: pc, ...AX },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.ruleX([0], { stroke: C.faint }),
      Plot.line([[lo, lo], [hi, hi]], { stroke: C.red, strokeWidth: 2, strokeDasharray: "5,4" }),
      Plot.link(scored, { x1: (d) => d.x + off(d), x2: (d) => d.x + off(d), y1: "lo", y2: "hi", stroke: (d) => S[d.seed], strokeOpacity: 0.45, strokeWidth: 1.2 }),
      Plot.dot(other, { x: (d) => d.x + off(d), y: "y", r: 4.5, stroke: C.muted, fill: "none", strokeWidth: 1.4, tip: true, title: tip }),
      Plot.dot(scored, { x: (d) => d.x + off(d), y: "y", r: 4, fill: (d) => S[d.seed], stroke: C.paper, strokeWidth: 0.8, tip: true, title: tip }),
      Plot.text(named, { ...halo(), x: "x", y: "y", text: (d) => (d.k === "A" ? T.name_A || "training probe" : T.name_emu || "rounding setting, no probe"), dx: -8, dy: -10, textAnchor: "end", fontSize: 12, fill: C.ink }),
    ],
  }), { style: `width:100%;height:auto;max-width:${W}px` }));
  if (T.note) root.append(el("font-size:12.5px;margin:0.3rem 0 0;color:" + C.muted, T.note));
  return root;
}
