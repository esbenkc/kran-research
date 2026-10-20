// Megaprobe vs spin lies: does more probe capacity rank "spin lies" above truthful answers?
// One panel. x = probe capacity, left to right: one layer read by a linear probe -> every layer's final token, an
// average of linear probes (public reference recipe) -> every layer and every answer token, attention pooling (ours)
// [-> the public axial attention design, if present]. y = AUROC of spin lies vs truthful answers that state the same
// fact (0.5 = coin flip, 1 = every spin lie ranked above every such truthful answer).
// Filled dot + solid line: probe fitted on all lie labels (what an auditor has). Hollow dot + dotted line: probe fitted
// only on answers that mention the fact (spin lies vs truthful). Vertical bar: 95% interval. Small dots: each trained
// model. Shaded strip: a words-only classifier on the same labels (95% interval). Dashed line: the 0.80 bar.
// data = {
//   title, y_label, band_note, bar,
//   arms: [{ key, label }],                                   // x order
//   series: [{ key, label, role }],                           // role "base" | "trained"
//   points: [{ series, arm, fit, auroc, lo, hi }],            // fit "all" | "fact"
//   per_model: [{ model, label, arm, fit, auroc }],           // trained models, fit "all"
//   words?: { auroc, lo, hi, label }
// }
import { C, FONT, plotStyle, gridY, halo } from "../theme.js";

const f2 = (v) => v.toFixed(2);

export function render(data, Plot) {
  const STYLE = {
    base: { color: C.ink, width: 2.2 },
    trained: { color: C.primary, width: 3 },
  };
  const series = data.series.map((s) => ({ ...s, ...STYLE[s.role] }));
  const meta = Object.fromEntries(series.map((s) => [s.key, s]));
  const armLabel = Object.fromEntries(data.arms.map((a) => [a.key, a.label]));
  const order = data.arms.map((a) => a.label);
  const pts = data.points
    .filter((p) => meta[p.series] && armLabel[p.arm] && p.auroc != null)
    .map((p) => ({ ...p, x: armLabel[p.arm] }));
  const per = (data.per_model || []).filter((p) => armLabel[p.arm] && p.auroc != null).map((p) => ({ ...p, x: armLabel[p.arm] }));

  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.4rem;color:" + C.ink;
  head.textContent = data.title;
  root.appendChild(head);

  const key = (svg, text, color) =>
    `<span style="display:inline-flex;align-items:center;gap:0.4rem${color ? ";color:" + color : ""}">${svg}${text}</span>`;
  const lineSvg = (color, width, dash) =>
    `<svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${color}" stroke-width="${width}"
      ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>`;
  const dotSvg = (color, hollow, r = 4.5) =>
    `<svg width="14" height="14"><circle cx="7" cy="7" r="${r}" fill="${hollow ? "none" : color}" stroke="${color}" stroke-width="1.8"/></svg>`;
  const legend = document.createElement("div");
  legend.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:0.35rem 1.2rem;font-size:15px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = series.map((s) => key(lineSvg(s.color, s.width + 0.4, null), s.label)).join("");
  root.appendChild(legend);

  const legend2 = document.createElement("div");
  legend2.style.cssText =
    "display:flex;flex-wrap:wrap;align-items:center;gap:0.3rem 1.1rem;font-size:13px;margin:0 0 0.4rem;color:" + C.muted;
  const parts = [
    key(dotSvg(C.muted, false), "filled dot: probe fitted on all lie labels"),
    key(dotSvg(C.muted, true), "hollow dot: probe fitted only on answers that mention the fact"),
  ];
  if (per.length) parts.push(key(dotSvg(C.muted, true, 2.5), "small dot: one trained model"));
  if (data.words)
    parts.push(
      key(`<svg width="26" height="12"><rect x="0" y="1" width="26" height="10" fill="${C.faint}" fill-opacity="0.45"/></svg>`,
        data.words.label),
    );
  if (data.bar != null) parts.push(key(lineSvg(C.red, 2.2, "3,3"), `the ${data.bar} bar`, C.red));
  legend2.innerHTML = parts.join("");
  root.appendChild(legend2);

  const marks = [gridY(Plot), Plot.ruleY([0.5], { stroke: C.faint, strokeDasharray: "1,3" })];
  if (data.words)
    marks.push(
      Plot.rect([data.words], { y1: "lo", y2: "hi", fill: C.faint, fillOpacity: 0.45 }), // no x: spans the frame
    );
  if (data.bar != null)
    marks.push(
      Plot.ruleY([data.bar], { stroke: C.red, strokeWidth: 2, strokeDasharray: "3,3" }),
    );

  // Each series has a small horizontal offset so its interval bars do not overlap the other series.
  const off = { base: -9, trained: 9 };
  const fitOff = { all: 0, fact: 16 };
  const dx = (d) => (off[meta[d.series]?.role] ?? 0) + (fitOff[d.fit] ?? 0);
  for (const s of series)
    for (const fit of ["all", "fact"]) {
      const line = order.map((x) => pts.find((p) => p.series === s.key && p.fit === fit && p.x === x)).filter(Boolean);
      marks.push(
        Plot.line(line, {
          x: "x", y: "auroc", dx: (off[s.role] ?? 0) + fitOff[fit], stroke: s.color,
          strokeWidth: fit === "all" ? s.width : 1.4, strokeDasharray: fit === "all" ? null : "2,3",
        }),
      );
    }
  const withCI = pts.filter((p) => p.lo != null);
  for (const s of series)
    for (const fit of ["all", "fact"]) {
      const sel = withCI.filter((p) => p.series === s.key && p.fit === fit);
      marks.push(
        Plot.ruleX(sel, {
          x: "x", y1: "lo", y2: "hi", dx: (off[s.role] ?? 0) + fitOff[fit], stroke: s.color, strokeWidth: 1.6,
          strokeOpacity: 0.85,
        }),
      );
    }
  if (per.length)
    marks.push(
      Plot.dot(per, {
        x: "x", y: "auroc", dx: off.trained - 6, r: 2.6, fill: "none", stroke: C.primary, strokeOpacity: 0.7,
        strokeWidth: 1.2, tip: true, title: (d) => `${d.label}, ${d.x}\nAUROC ${f2(d.auroc)}`,
      }),
    );
  for (const s of series)
    for (const fit of ["all", "fact"]) {
      const sel = pts.filter((p) => p.series === s.key && p.fit === fit);
      marks.push(
        Plot.dot(sel, {
          x: "x", y: "auroc", dx: (off[s.role] ?? 0) + fitOff[fit], r: 5, stroke: s.color, strokeWidth: 1.8,
          fill: fit === "all" ? s.color : C.paper, tip: true,
          title: (d) =>
            `${s.label}, ${d.x}\n${fit === "all" ? "fitted on all lie labels" : "fitted only on answers that mention the fact"}` +
            `\nAUROC ${f2(d.auroc)}` + (d.lo != null ? ` (95% interval ${f2(d.lo)}–${f2(d.hi)})` : ""),
        }),
      );
    }

  const lo = Math.min(0.5, ...pts.map((p) => p.lo ?? p.auroc), ...per.map((p) => p.auroc), ...(data.words ? [data.words.lo] : []));
  const hi = Math.max(0.85, ...pts.map((p) => p.hi ?? p.auroc), ...per.map((p) => p.auroc));
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.2rem 48px;color:" + C.muted;
  sub.textContent = data.y_label;
  root.appendChild(sub);
  root.appendChild(
    Plot.plot({
      style: plotStyle({ fontSize: "14px" }),
      width: 640, height: 420, marginTop: 16, marginLeft: 44, marginRight: 40, marginBottom: 70,
      x: { type: "point", domain: order, label: null, padding: 0.35 },
      y: { domain: [Math.floor(lo * 20) / 20, Math.min(1, Math.ceil(hi * 20) / 20)], label: null, ticks: 6 },
      marks: [
        ...marks,
        Plot.axisX({ lineWidth: 12, fontSize: 13, tickSize: 0 }),
      ],
    }),
  );
  if (data.band_note) {
    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;margin:0.2rem 0 0 48px;color:" + C.muted;
    note.textContent = data.band_note;
    root.appendChild(note);
  }
  return root;
}
