// One row per way of running Qwen3-1.7B on an Apple M4 (bf16). Left: largest change in an output score when the
// probe is attached (log axis; hollow dot at the left edge = every bit identical). Right: share of right answers
// on 128 new prompts x 2 (probe on / off), with the trained last two layers put back into the whole model.
// data = { headline, title, subtitle, floor, chance, bar, rows:[{label, check, diff, changed, n, acc?, nulls?, note?}] }
import { C, FONT, plotStyle, halo } from "../theme.js";

const pc = (v) => `${Math.round(v * 1000) / 10}%`;

export function render(data, Plot) {
  const rows = data.rows;
  const order = rows.map((r) => r.label);
  const col = (r) => (r.check ? C.secondary : C.primary);
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const add = (css, txt) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = txt;
    root.appendChild(d);
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);

  const W = Math.max(560, Math.min(900, root.clientWidth || 860));
  const H = 46 * rows.length + 70;
  const y = { domain: order, label: null, tickSize: 0 };
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:wrap;gap:0 18px;align-items:flex-start";
  root.appendChild(wrap);

  const same = rows.filter((r) => !(r.diff > 0));
  const diff = rows.filter((r) => r.diff > 0);
  const left = Plot.plot({
    style: plotStyle({ fontSize: "14px" }),
    width: Math.round(W * 0.56), height: H, marginLeft: 210, marginRight: 76, marginTop: 30, marginBottom: 44,
    x: { type: "log", domain: [data.floor * 0.8, 4], label: "Largest change in an output score (log)", labelAnchor: "center", labelArrow: "none" },
    y,
    marks: [
      Plot.gridX({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.dot(same, {
        x: () => data.floor, y: "label", r: 6, stroke: C.muted, strokeWidth: 1.8, fill: "none", tip: true,
        title: (r) => `${r.label}\nevery bit of every score identical on ${r.n} of ${r.n} prompts`,
      }),
      Plot.text(same, { x: () => data.floor, y: "label", text: () => "bit-identical", dx: 10, textAnchor: "start", fill: C.muted, fontSize: 12 }),
      Plot.dot(diff, {
        x: "diff", y: "label", r: 6.5, fill: col, tip: true,
        title: (r) => `${r.label}\nlargest change in a score: ${r.diff}\nprompts with any change: ${r.changed} of ${r.n}`,
      }),
      Plot.text(diff, { x: "diff", y: "label", text: (r) => `${r.changed}/${r.n} prompts`, dx: 10, textAnchor: "start", fill: C.ink, fontSize: 12, ...halo() }),
    ],
  });

  const pts = [];
  const nul = [];
  for (const r of rows) {
    (r.acc ?? []).forEach((a, s) => pts.push({ ...r, a, s }));
    for (const n of r.nulls ?? []) n.acc.forEach((a, s) => nul.push({ label: r.label, kind: n.kind, a, s }));
  }
  const notes = rows.filter((r) => r.note);
  const right = Plot.plot({
    style: plotStyle({ fontSize: "14px" }),
    width: Math.round(W * 0.40), height: H, marginLeft: 12, marginRight: 16, marginTop: 30, marginBottom: 44,
    x: { domain: [0.3, 1.02], ticks: [0.4, 0.5, 0.6, 0.75, 0.9, 1], tickFormat: (v) => `${Math.round(v * 100)}%`, label: "Detector right on new prompts", labelAnchor: "center", labelArrow: "none" },
    y: { ...y, axis: null },
    marks: [
      Plot.ruleX([data.chance], { stroke: C.faint, strokeDasharray: "4,4", strokeWidth: 1.4 }),
      Plot.ruleX([data.bar], { stroke: C.red, strokeWidth: 1.4 }),
      Plot.text([data.chance], { x: (v) => v, y: () => order[0], text: () => "chance", dy: -24, fill: C.muted, fontSize: 12 }),
      Plot.text([data.bar], { x: (v) => v, y: () => order[0], text: () => "pass bar", dy: -24, fill: C.red, fontSize: 12 }),
      Plot.dot(nul, {
        x: "a", y: "label", r: 5, stroke: C.ink, strokeWidth: 1.4, fill: "none", symbol: (d) => (d.kind === "shuffled" ? "circle" : "square"),
        dy: 9, tip: true, title: (d) => `${d.label}\ncontrol: ${d.kind === "shuffled" ? "answers shuffled" : "probe never attached"}, seed ${d.s}\nright: ${pc(d.a)}`,
      }),
      Plot.dot(pts, {
        x: "a", y: "label", r: 6.5, fill: col, stroke: C.paper, strokeWidth: 1, dy: -4, tip: true,
        title: (d) => `${d.label}\nseed ${d.s}: right on ${pc(d.a)} of 256 answers`,
      }),
      Plot.text(notes, { x: () => 0.31, y: "label", text: "note", textAnchor: "start", fill: C.muted, fontSize: 12 }),
    ],
  });
  wrap.append(left, right);

  const key = document.createElement("div");
  key.style.cssText = `font-size:13px;color:${C.muted};margin-top:0.2rem`;
  key.textContent = "Filled dots: one per training seed. Hollow circle: answers shuffled in training. Hollow square: probe never attached in training (both answers saw the same numbers).";
  root.appendChild(key);
  return root;
}
