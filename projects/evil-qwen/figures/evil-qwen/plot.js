// Does lyra's two-neuron push make Qwen3.5-4B evil? One row per setting, one bar per judge.
// Each bar = 44 answers sorted into four kinds: evil (wants harm, judge score >= 50) x on topic (coherence >= 50).
// data = { headline, rows: [{label, cells_by_judge: {judge: {harm_coh, harm_broken, noharm_broken, noharm_coh}}}], n }
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));
  const h = data.headline;
  const JN = data.n.judges;
  const J = Object.keys(JN);
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "One neuron makes Qwen talk like a villain. Random neurons only break it.";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  const byJ = (o) => J.map((j) => `${pct(o[j])} (${JN[j]})`).join(", ");
  const FN = data.n.judge_full;
  sub.textContent = `Each bar = ${data.n.answers_per_bar} answers, sorted into four kinds by one judge ` +
    `(upper bar ${FN[J[0]]}, lower bar ${FN[J[1]]}). With lyra's push, ${byJ(h.lyra_evil_by_judge)} of answers are evil, ` +
    `but only ${byJ(h.lyra_evil_on_topic_by_judge)} also stay on topic. ` +
    `The same push on ${h.random_pairs} random neuron pairs: at most ${byJ(h.random_evil_max_by_judge)} evil in any pair.`;
  const K = [
    { c: "harm_coh", name: "evil, on topic", fill: C.primary },
    { c: "harm_broken", name: "evil, falls apart", fill: C.primaryLight },
    { c: "noharm_broken", name: "falls apart", fill: C.neutral },
    { c: "noharm_coh", name: "normal answer", fill: C.secondary },
  ];
  const sw = (k) => `<svg width="14" height="14"><rect width="14" height="14" rx="2" fill="${k.fill}"/></svg>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = K.map((k) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${sw(k)}${k.name}</span>`).join("");
  root.append(head, sub, legend);

  const order = K.map((k) => k.name);
  const rows = data.rows.flatMap((r) => J.map((j) => [j, r.cells_by_judge[j]]).flatMap(([j, cells]) =>
    K.map((k) => ({ y: r.label, j: JN[j], k: k.name, v: cells[k.c] }))));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 470,
    marginLeft: 250, marginRight: 58, marginTop: 6, marginBottom: 40,
    x: { domain: [0, 1], label: "Share of answers", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    fy: { domain: data.rows.map((r) => r.label), label: null, tickSize: 0, axis: "left" },
    y: { domain: J.map((j) => JN[j]), label: null, tickSize: 0, axis: "right" },
    color: { domain: order, range: K.map((k) => k.fill) },
    marks: [
      Plot.barX(rows, Plot.stackX({ fy: "y", y: "j", x: "v", fill: "k", order, tip: true,
        title: (r) => `${r.y}\n${r.j}: ${r.k} ${pct(r.v)}` })),
      Plot.text(rows.filter((r) => r.v >= 0.05), Plot.stackX({ fy: "y", y: "j", x: "v", z: "k", order,
        text: (r) => pct(r.v), fill: C.ink, ...halo(), fontSize: 11 })),
    ],
  }));
  return root;
}
