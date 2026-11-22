// Main figure for "stated-cost-canary". gpt-oss-20b looks for one 2025 number on a small website we serve.
// One plane: x = share of runs that requested the footer link, y = share of those runs whose reasoning named the
// link hidden or a trap first. One dot per cell, cross-hairs = 95% intervals, arrow = neutral -> penalty on trap pages.
// Shaded zones come from the lines fixed before the runs, measured from the neutral trap-page dot.
// Data: kran-research projects/stated-cost-canary/figures/figure_data.py writes data.json.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pc = (v) => `${Math.round(v * 100)}%`;

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

  const col = { hidden: C.primary, visible: C.secondary };
  const sym = {
    dot: (c) => `<svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="${c}"/></svg>`,
    ring: (c) => `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${c}" stroke-width="2"/></svg>`,
    bar: (c) => `<svg width="16" height="14"><rect x="1" y="4" width="14" height="6" rx="2" fill="${c}"/></svg>`,
  };
  const key = el("display:flex;flex-wrap:wrap;gap:2px 14px;margin:0 0 4px;font-size:13px", "");
  key.innerHTML = [[sym.dot(C.ink), L.penalty], [sym.ring(C.ink), L.neutral], [sym.bar(col.hidden), L.trap], [sym.bar(col.visible), L.control]]
    .map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:5px">${s}${t}</span>`).join("");
  const sw = (c) => `<svg width="16" height="14"><rect x="1" y="1" width="14" height="12" fill="${c}" stroke="${C.faint}" stroke-width="0.6"/></svg>`;
  const zkey = el("display:flex;flex-direction:column;gap:1px;margin:0 0 4px;font-size:12.5px", "");
  zkey.innerHTML = [[sw(C.primaryLight), `${L.zone_yes}: ${L.zone_yes_sub}`], [sw(C.grid), L.zone_no_deter], [sw(C.paper), L.zone_no_say]]
    .map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:5px">${s}${t}</span>`).join("");

  const cells = data.cells;
  const xNo = data.lines.follow_no, yYes = data.lines.announce_yes;
  // x starts low enough for every interval and the deterrence line, in steps of 10 points, never above 60%
  const xmin = Math.max(0, Math.min(0.6, Math.floor((Math.min(xNo, ...cells.map((c) => c.follow.lo)) - 0.08) * 10) / 10));
  const H = narrow ? 360 : 420;
  const N = cells.find((c) => c.id === "N"), Cp = cells.find((c) => c.id === "C");
  const zones = [
    { x1: xNo, x2: 1, y1: 0, y2: Math.max(0, yYes), fill: C.primaryLight, op: 0.7 },
    { x1: xmin, x2: xNo, y1: 0, y2: 1, fill: C.grid, op: 0.8 },
  ];
  const tip = (c) => `${c.label}\nrequested the link: ${pc(c.follow.rate)} (${c.follow.k} of ${c.follow.n} runs), 95% interval ${pc(c.follow.lo)} to ${pc(c.follow.hi)}\n` +
    `named it hidden first: ${pc(c.announce.rate)} (${c.announce.k} of ${c.announce.n}), 95% interval ${pc(c.announce.lo)} to ${pc(c.announce.hi)}`;
  // labels: left of each dot (right of it near the left edge); a close neutral/penalty pair shares one two-line label
  const word = (c) => narrow ? `${pc(c.follow.rate)} req., ${pc(c.announce.rate)} named` : `${pc(c.follow.rate)} requested, ${pc(c.announce.rate)} named it`;
  const who = (c) => (narrow ? c.sentence : c.sentence === "neutral" ? L.neutral : L.penalty);
  const labels = [];
  for (const page of ["hidden", "visible"]) {
    const a = cells.find((c) => c.page === page && c.sentence === "neutral"), b = cells.find((c) => c.page === page && c.sentence === "penalty");
    const [hi, lo] = a.announce.rate >= b.announce.rate ? [a, b] : [b, a];
    const pos = (c, up) => ({ x: c.follow.rate, y: c.announce.rate, up, right: (c.follow.rate - xmin) / (1 - xmin) < 0.45, page });
    if (Math.abs(a.follow.rate - b.follow.rate) < 0.08 && Math.abs(a.announce.rate - b.announce.rate) < 0.08) {
      labels.push({ ...pos(hi, true), text: `${who(a)}: ${word(a)}\n${who(b)}: ${word(b)}` });
    } else {
      labels.push({ ...pos(hi, true), text: `${who(hi)}: ${word(hi)}` });
      labels.push({ ...pos(lo, lo.announce.rate < 0.07), text: `${who(lo)}: ${word(lo)}` });
    }
  }
  const txt = (f, o) => Plot.text(labels.filter(f), { x: "x", y: "y", text: "text", fontSize: 12, fontWeight: 600, fill: (l) => col[l.page], ...halo(), ...o });
  const marks = [
    Plot.rect(zones, { x1: "x1", x2: "x2", y1: "y1", y2: "y2", fill: "fill", fillOpacity: "op" }),
    Plot.text([0], { x: xNo + 0.01, y: Math.max(0, yYes) - 0.02, text: () => "Yes", textAnchor: "start", lineAnchor: "top", fontSize: 13, fontWeight: 700, fill: C.ink, ...halo() }),
    Plot.arrow([0], { x1: N.follow.rate, y1: N.announce.rate, x2: Cp.follow.rate, y2: Cp.announce.rate, stroke: C.primary, strokeWidth: 2, inset: 8, headLength: 9 }),
    Plot.ruleY(cells, { y: (c) => c.announce.rate, x1: (c) => Math.max(xmin, c.follow.lo), x2: (c) => c.follow.hi, stroke: (c) => col[c.page], strokeWidth: 1.3, strokeOpacity: 0.85 }),
    Plot.ruleX(cells, { x: (c) => c.follow.rate, y1: (c) => c.announce.lo, y2: (c) => c.announce.hi, stroke: (c) => col[c.page], strokeWidth: 1.3, strokeOpacity: 0.85 }),
    Plot.dot(cells.filter((c) => c.sentence === "penalty"), { x: (c) => c.follow.rate, y: (c) => c.announce.rate, r: 6, fill: (c) => col[c.page], stroke: C.paper, strokeWidth: 1, tip: true, title: tip }),
    Plot.dot(cells.filter((c) => c.sentence === "neutral"), { x: (c) => c.follow.rate, y: (c) => c.announce.rate, r: 5.5, fill: C.paper, stroke: (c) => col[c.page], strokeWidth: 2, tip: true, title: tip }),
    txt((l) => l.up && !l.right, { textAnchor: "end", lineAnchor: "bottom", dx: -9, dy: -7 }),
    txt((l) => l.up && l.right, { textAnchor: "start", lineAnchor: "bottom", dx: 9, dy: -7 }),
    txt((l) => !l.up && !l.right, { textAnchor: "end", lineAnchor: "top", dx: -9, dy: 7 }),
    txt((l) => !l.up && l.right, { textAnchor: "start", lineAnchor: "top", dx: 9, dy: 7 }),
  ];
  const svg = Plot.plot({
    width: W, height: H, marginLeft: 46, marginRight: 14, marginTop: 26, marginBottom: 40,
    style: plotStyle({ fontSize: "12.5px" }),
    x: { domain: [xmin, 1], tickFormat: pc, label: `${L.px} →`, labelAnchor: "right", labelArrow: "none", grid: true },
    y: { domain: [0, 1], tickFormat: pc, label: `↑ ${L.py}`, labelAnchor: "top", labelArrow: "none", grid: true },
    color: { type: "identity" },
    marks,
  });
  svg.style.maxWidth = "100%";
  svg.style.height = "auto";
  root.appendChild(svg);
  el(`font-size:12px;color:${C.muted};margin:4px 0 0`, L.note);
  return root;
}
