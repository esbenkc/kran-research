// Main figure for "stated-cost-canary". gpt-oss-20b looks for one 2025 number on a small website we serve.
// Panels: trap pages (footer link hidden in the HTML) and control pages (same link shown). Rows: share of runs
// that requested the link, and share of those runs whose reasoning named the link hidden or a trap first.
// Filled dot = penalty sentence, hollow ring = neutral sentence, bars = 95% intervals over runs.
// Dashed tick = the line fixed before the runs, measured from the neutral dot. Data: figures/figure_data.py.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pc = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const L = data.legend;
  const vw = typeof window !== "undefined" ? window.innerWidth : 600;
  const W = Math.max(320, Math.min(556, vw - 40));
  const narrow = W < 460;
  const ML = narrow ? 120 : 178;
  const root = document.createElement("div");
  root.style.cssText = `font-family:${FONT};text-align:left;max-width:${W}px;color:${C.ink}`;
  const el = (css, text, parent = root) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    if (text) d.textContent = text;
    parent.appendChild(d);
    return d;
  };

  el("font-size:17px;font-weight:600;margin:0 0 2px", data.title);
  el(`font-size:14px;margin:0 0 6px;color:${C.muted}`, data.subtitle);
  if (data.phase !== "counted") el(`font-size:13px;font-weight:700;color:${C.red}`, `NOT COUNTED (${data.phase} data): layout test only`);

  const col = { hidden: C.primary, visible: C.secondary };
  const sym = {
    dot: (c) => `<svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="${c}"/></svg>`,
    ring: (c) => `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${c}" stroke-width="2"/></svg>`,
    tick: (c) => `<svg width="10" height="16"><line x1="5" y1="1" x2="5" y2="15" stroke="${c}" stroke-width="2" stroke-dasharray="3,2"/></svg>`,
  };
  const key = el("display:flex;flex-wrap:wrap;gap:2px 14px;margin:0 0 4px;font-size:13px", "");
  key.innerHTML = [[sym.dot(C.ink), L.penalty], [sym.ring(C.ink), L.neutral], [sym.tick(C.red), L.line]]
    .map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:5px">${s}${t}</span>`).join("");

  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const label = (r) => (narrow ? r.short : r.label);

  const panel = (group, last) => {
    const rows = data.rows.filter((r) => r.group === group.id);
    const c = col[group.page];
    const box = el("", "");
    el(`font-size:13.5px;font-weight:600;margin:10px 0 0;color:${C.ink}`, group.label, box);
    if (group.sub) el(`font-size:12.5px;margin:0 0 2px;color:${C.muted}`, group.sub, box);
    const pending = rows.find((r) => r.pending);
    if (pending) {
      el(`font-size:12.5px;margin:4px 0 0 ${ML}px;padding:3px 8px;border:1px dashed ${C.grid};border-radius:4px;color:${C.ink};display:inline-block`, pending.note, box);
      return;
    }
    const ys = rows.map(label);
    const pts = rows.flatMap((r) => [
      { ...r.p, row: label(r), s: "penalty", r },
      { ...r.n, row: label(r), s: "neutral", r },
    ]);
    const tip = (p) => `${p.r.label}, ${p.s === "penalty" ? L.penalty : L.neutral}: ${pc(p.rate)} (${p.k} of ${p.n} runs)\n` +
      `95% interval ${pc(p.lo)} to ${pc(p.hi)}` + (p.seed_lo != null ? `\nseed-level ${pc(p.seed_lo)} to ${pc(p.seed_hi)}` : "");
    // value labels just past the outer end of each interval; one shared label when the two dots nearly touch
    const labels = rows.flatMap((r) => {
      const lo = r.p.rate <= r.n.rate ? r.p : r.n, hi = lo === r.p ? r.n : r.p;
      const left = Math.min(r.p.lo, r.n.lo), right = Math.max(r.p.hi, r.n.hi);
      const both = Math.abs(r.p.rate - r.n.rate) < 0.005 ? `both ${pc(r.p.rate)}` : `${pc(lo.rate)} / ${pc(hi.rate)}`;
      if (left < 0.12) return [{ row: label(r), x: right, text: both, side: "right" }];
      if (hi.rate - lo.rate < 0.1) return [{ row: label(r), x: right > 0.8 ? left : right, text: both, side: right > 0.8 ? "left" : "right" }];
      return [{ row: label(r), x: left, text: pc(lo.rate), side: "left" }, { row: label(r), x: right, text: pc(hi.rate), side: "right" }];
    });
    const lines = rows.filter((r) => r.line != null);
    const marks = [
      Plot.gridX({ stroke: C.grid, strokeOpacity: 1, ticks }),
      Plot.tickX(lines, { x: "line", y: label, stroke: C.red, strokeWidth: 2, strokeDasharray: "3,2", inset: 2 }),
      Plot.text(lines, { x: "line", y: label, text: "line_label", dy: -15, fontSize: 11, fill: C.red, ...halo() }),
      Plot.link(rows, { y1: label, y2: label, x1: (r) => r.p.rate, x2: (r) => r.n.rate, stroke: c, strokeWidth: 2.5, strokeOpacity: 0.35 }),
    ];
    for (const s of ["penalty", "neutral"]) {
      const ps = pts.filter((p) => p.s === s);
      const dy = s === "penalty" ? -5 : 5;
      marks.push(Plot.ruleY(ps, { y: "row", x1: "lo", x2: "hi", stroke: c, strokeWidth: 1.3, dy, strokeOpacity: 0.85 }));
      marks.push(s === "penalty"
        ? Plot.dot(ps, { y: "row", x: "rate", r: 5.5, fill: c, stroke: C.paper, strokeWidth: 1, tip: true, title: tip })
        : Plot.dot(ps, { y: "row", x: "rate", r: 5, fill: C.paper, stroke: c, strokeWidth: 2, tip: true, title: tip }));
    }
    marks.push(Plot.text(labels.filter((l) => l.side === "left"), { y: "row", x: "x", text: "text", textAnchor: "end", dx: -10, fontSize: 12, fontWeight: 600, fill: c, ...halo() }));
    marks.push(Plot.text(labels.filter((l) => l.side === "right"), { y: "row", x: "x", text: "text", textAnchor: "start", dx: 10, fontSize: 12, fontWeight: 600, fill: c, ...halo() }));
    const svg = Plot.plot({
      width: W, height: rows.length * 46 + 22 + (last ? 36 : 10), marginLeft: ML, marginRight: 30, marginTop: 22, marginBottom: last ? 36 : 10,
      style: plotStyle({ fontSize: "12.5px" }),
      x: { domain: [0, 1], ticks, tickFormat: pc, label: last ? L.x : null, labelAnchor: "center", labelArrow: "none", axis: last ? "bottom" : null },
      y: { domain: ys, label: null, tickSize: 0, inset: 8 },
      marks,
    });
    svg.style.maxWidth = "100%";
    svg.style.height = "auto";
    box.appendChild(svg);
  };
  data.groups.forEach((g, i) => panel(g, i === data.groups.length - 1));
  el(`font-size:12px;color:${C.muted};margin:4px 0 0`, L.note);
  return root;
}
