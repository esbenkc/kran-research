// Main figure for "stated-cost-canary". gpt-oss-20b looks for one 2025 number on a small website we serve.
// Slope chart, trap pages: neutral sentence (left) -> penalty sentence (right). Line 1: share of runs that requested
// the hidden link. Line 2: share of those runs whose reasoning named the link hidden or a trap first. Faint line: the
// same words on control pages, where nothing is hidden. Whiskers: 95% intervals over runs. Dashed ticks at the penalty
// column: the lines fixed before the runs. Data: kran-research projects/stated-cost-canary/figures/figure_data.py.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pc = (v) => `${Math.round(v * 100)}%`;

// spread label y positions (data units) so no two sit closer than gap; keeps the order of the values
function dodge(items, gap) {
  const s = [...items].sort((a, b) => b.v - a.v);
  if (s.length) s[0].ly = s[0].v;
  for (let i = 1; i < s.length; i++) s[i].ly = Math.min(s[i].v, s[i - 1].ly - gap);
  const low = s.length ? s[s.length - 1].ly : 0;
  if (low < 0.0) s.forEach((d) => (d.ly -= low));
  return s;
}

export function render(data, Plot) {
  const L = data.legend;
  const vw = typeof window !== "undefined" ? window.innerWidth : 600;
  const W = Math.max(320, Math.min(556, vw - 40));
  const narrow = W < 460;
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

  const cell = Object.fromEntries(data.cells.map((c) => [c.id, c]));
  const series = [
    { id: "req", name: L.s_req, short: L.s_req_short, col: C.secondary, n: cell.N.follow, p: cell.C.follow, w: 3 },
    { id: "say", name: L.s_say, short: L.s_say_short, col: C.primary, n: cell.N.announce, p: cell.C.announce, w: 3 },
    { id: "echo", name: L.s_echo, short: L.s_echo_short, col: C.faint, n: cell.NV.announce, p: cell.CV.announce, w: 1.6 },
  ];
  // numeric x: 0 = neutral sentence, 1 = penalty sentence, so the frozen ticks can be short rules at x = 1
  const X = [0, 1];
  const xname = narrow ? [L.col_neutral_short, L.col_penalty_short] : [L.col_neutral, L.col_penalty];
  const pts = series.flatMap((s) => [{ s, x: 0, col: xname[0], ...s.n }, { s, x: 1, col: xname[1], ...s.p }]);
  const ticks = [
    { y: data.lines.follow_no, col: C.secondary },
    { y: data.lines.announce_yes, col: C.primary },
  ].filter((t) => t.y > 0);
  const tip = (d) => `${d.s.name}, ${d.col}: ${pc(d.rate)} (${d.k} of ${d.n} runs)\n95% interval ${pc(d.lo)} to ${pc(d.hi)}` +
    (d.seed_lo != null ? `\nseed-level ${pc(d.seed_lo)} to ${pc(d.seed_hi)}` : "");
  const right = dodge(series.map((s) => ({ v: s.p.rate, s, text: narrow ? `${pc(s.p.rate)} ${s.short}\n${s.p.k} of ${s.p.n}` : `${pc(s.p.rate)} ${s.name}\n${s.p.k} of ${s.p.n} runs` })), narrow ? 0.09 : 0.1);
  const left = dodge(series.map((s) => ({ v: s.n.rate, s, text: pc(s.n.rate) })), 0.06);

  const key = el("display:flex;flex-direction:column;gap:1px;margin:0 0 4px;font-size:12.5px", "");
  const dash = `<svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" stroke="${C.ink}" stroke-width="2" stroke-dasharray="4,2"/></svg>`;
  key.innerHTML = `<span style="display:inline-flex;align-items:center;gap:5px">${dash}${L.ticks}</span>`;

  const H = narrow ? 340 : 380;
  const svg = Plot.plot({
    width: W, height: H, marginLeft: 44, marginRight: narrow ? 128 : 236, marginTop: 14, marginBottom: 30,
    style: plotStyle({ fontSize: "12.5px" }),
    x: { domain: [narrow ? -0.45 : -0.3, 1.12], ticks: X, tickFormat: (v) => xname[v], label: null, tickSize: 0 },
    y: { domain: [0, 1], tickFormat: pc, label: null, grid: true, ticks: [0, 0.25, 0.5, 0.75, 1] },
    color: { type: "identity" },
    marks: [
      Plot.line(pts, { x: "x", y: "rate", z: (d) => d.s.id, stroke: (d) => d.s.col, strokeWidth: (d) => d.s.w }),
      Plot.ruleX(pts, { x: "x", y1: "lo", y2: "hi", stroke: (d) => d.s.col, strokeWidth: 1.4 }),
      Plot.ruleY(ticks, { x1: 0.9, x2: 1.1, y: "y", stroke: "col", strokeWidth: 2.2, strokeDasharray: "4,2" }),
      Plot.dot(pts, { x: "x", y: "rate", r: (d) => (d.s.id === "echo" ? 4 : 5.5), fill: (d) => (d.s.id === "echo" ? C.paper : d.s.col), stroke: (d) => d.s.col, strokeWidth: 1.6, tip: true, title: tip }),
      Plot.text(right, { x: () => 1, y: "ly", text: "text", textAnchor: "start", dx: 26, fontSize: 12.5, fontWeight: 600, fill: (d) => (d.s.id === "echo" ? C.muted : d.s.col), ...halo() }),
      Plot.text(left, { x: () => 0, y: "ly", text: "text", textAnchor: "end", dx: -12, fontSize: 12.5, fontWeight: 600, fill: (d) => (d.s.id === "echo" ? C.muted : d.s.col), ...halo() }),
    ],
  });
  svg.style.maxWidth = "100%";
  svg.style.height = "auto";
  root.appendChild(svg);
  el(`font-size:12px;color:${C.muted};margin:4px 0 0`, L.note);
  return root;
}
