// V3: what happens to each answer as the push grows. 100% stacked bars per push size, plus random pairs.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${Math.round(v * 100)}%`;
const LAB = { 0: "0 (normal)", 0.25: "¼", 0.5: "½", 0.75: "¾", 1: "1 (lyra)", 1.5: "1½", 2: "2", 3: "3", 4: "4" };

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const h = data.headline;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "Lyra's push turns Qwen into a villain that loses the thread";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  const rm = data.rand_mean["1.0"];
  sub.textContent = `Each bar = 44 answers, sorted by two judges into four kinds. At lyra's dose, ${pct(h.lyra_harm)} say the model wants harm. ` +
    `The same push on ${rm.n_pairs} random neuron pairs: ${pct(rm.cells.harm_coh + rm.cells.harm_broken)} want harm, ${pct(rm.cells.noharm_broken)} just break.`;
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
  const cols = data.dose.map((d) => ({ x: LAB[d.dose], cells: d.cells, n: d.n }));
  cols.push({ x: "random pairs at 1", cells: rm.cells, n: rm.n_pairs * 44 });
  const rows = cols.flatMap((col) => K.map((k) => ({ x: col.x, k: k.name, v: col.cells[k.c], fill: k.fill, n: col.n })));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 340,
    marginLeft: 48, marginRight: 10, marginTop: 10, marginBottom: 44,
    x: { domain: cols.map((c) => c.x), label: "Push size (1 = lyra's dose)", labelAnchor: "center", labelArrow: "none", padding: 0.15 },
    y: { domain: [0, 1], label: "Share of answers", tickFormat: (v) => `${Math.round(v * 100)}%`, labelArrow: "none" },
    color: { domain: K.map((k) => k.name), range: K.map((k) => k.fill) },
    marks: [
      Plot.barY(rows, Plot.stackY({ x: "x", y: "v", fill: "k", order: K.map((k) => k.name), tip: true,
        title: (r) => `${r.x}\n${r.k}: ${pct(r.v)}\n(n = ${r.n} answers, mean of two judges)` })),
      Plot.text(rows.filter((r) => r.v >= 0.12), Plot.stackY({ x: "x", y: "v", z: "k", order: K.map((k) => k.name), text: (r) => pct(r.v), fill: C.ink, ...halo(), fontSize: 11 })),
    ],
  }));
  return root;
}
