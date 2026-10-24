// V6: which neuron does it? Per setting, one stacked bar per judge (Claude Haiku 4.5, Gemini Flash).
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
  sub.textContent = "Each bar = 44 answers, sorted into four kinds by one judge (top: Claude Haiku, bottom: Gemini Flash). Push = lyra's dose unless noted.";
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
  const d34 = data.dose.find((d) => d.dose === 0.75);
  const bars = [
    { y: "normal Qwen (no push)", cj: d0.cells_by_judge },
    { y: "lyra's pair at ¾ push", cj: d34.cells_by_judge },
    { y: "lyra's pair: +L14 N8949, −L13 N7956", cj: d1.cells_by_judge },
    { y: "only +L14 N8949", cj: data.singles["only_L14N8949+1"].cells_by_judge },
    { y: "only −L13 N7956", cj: data.singles["only_L13N7956-1"].cells_by_judge },
    { y: `${rm.n_pairs} random pairs (mean)`, cj: rm.cells_by_judge },
  ];
  const JN = { "claude-haiku-4-5": "Haiku", "gemini-flash-latest": "Gemini" };
  const rows = bars.flatMap((b) => Object.entries(b.cj).flatMap(([j, cells]) =>
    K.map((k) => ({ y: b.y, j: JN[j], k: k.name, v: cells[k.c] }))));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 400,
    marginLeft: 300, marginRight: 58, marginTop: 6, marginBottom: 40,
    fy: { domain: bars.map((b) => b.y), label: null, tickSize: 0, axis: "left" },
    x: { domain: [0, 1], label: "Share of answers", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: ["Haiku", "Gemini"], label: null, tickSize: 0, axis: "right", tickPadding: -2 },
    color: { domain: K.map((k) => k.name), range: K.map((k) => k.fill) },
    marks: [
      Plot.barX(rows, Plot.stackX({ fy: "y", y: "j", x: "v", fill: "k", order: K.map((k) => k.name), tip: true, title: (r) => `${r.y} (${r.j})\n${r.k}: ${pct(r.v)}` })),
      Plot.text(rows.filter((r) => r.v >= 0.05), Plot.stackX({ fy: "y", y: "j", x: "v", z: "k", order: K.map((k) => k.name), text: (r) => pct(r.v), fill: C.ink, ...halo(), fontSize: 11 })),
    ],
  }));
  return root;
}
