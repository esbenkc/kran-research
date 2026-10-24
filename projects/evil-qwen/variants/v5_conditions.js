// V5: which neuron does it? One stacked bar per setting at lyra's dose (1x).
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "One neuron makes Qwen want harm; random neurons only break it";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = "Each bar = 44 answers at lyra's push size, sorted by two judges into four kinds (mean of the judges).";
  const K = [
    { c: "harm_coh", name: "wants harm, on topic", fill: C.primary },
    { c: "harm_broken", name: "wants harm, falls apart", fill: C.primaryLight },
    { c: "noharm_broken", name: "falls apart", fill: C.neutral },
    { c: "noharm_coh", name: "normal answer", fill: C.secondary },
  ];
  const sw = (k) => `<svg width="14" height="14"><rect width="14" height="14" rx="2" fill="${k.fill}"/></svg>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = K.map((k) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${sw(k)}${k.name}</span>`).join("");
  root.append(head, sub, legend);
  const d1 = data.dose.find((d) => d.dose === 1);
  const d0 = data.dose.find((d) => d.dose === 0);
  const rm = data.rand_mean["1.0"];
  const bars = [
    { y: "normal Qwen (no push)", cells: d0.cells },
    { y: "lyra's pair: +L14 N8949, −L13 N7956", cells: d1.cells },
    { y: "only +L14 N8949", cells: data.singles["only_L14N8949+1"].cells },
    { y: "only −L13 N7956", cells: data.singles["only_L13N7956-1"].cells },
    { y: "both signs flipped", cells: data.singles["flip_both"].cells },
    { y: `${rm.n_pairs} random pairs (mean)`, cells: rm.cells },
  ];
  const rows = bars.flatMap((b) => K.map((k) => ({ y: b.y, k: k.name, v: b.cells[k.c] })));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 280,
    marginLeft: 250, marginRight: 14, marginTop: 6, marginBottom: 40,
    x: { domain: [0, 1], label: "Share of answers", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: bars.map((b) => b.y), label: null, tickSize: 0 },
    color: { domain: K.map((k) => k.name), range: K.map((k) => k.fill) },
    marks: [
      Plot.barX(rows, Plot.stackX({ y: "y", x: "v", fill: "k", order: K.map((k) => k.name), tip: true, title: (r) => `${r.y}\n${r.k}: ${pct(r.v)}` })),
      Plot.text(rows.filter((r) => r.v >= 0.08), Plot.stackX({ y: "y", x: "v", z: "k", order: K.map((k) => k.name), text: (r) => pct(r.v), fill: C.ink, ...halo(), fontSize: 11 })),
    ],
  }));
  return root;
}
