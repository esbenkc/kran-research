// ─────────────────────────────────────────────────────────────────────────
//  Kran Research — figure style guide
//  Single source of truth for every interactive figure on blog.kran.ai.
//  Import this from any plot.js:  import { C, FONT, plotStyle, gridY } from "../theme.js";
//  Palette is drawn from the site's Flexoki theme. Ground = warm paper, type =
//  Inter, series = calm blue (primary) + amber (secondary).
//  Rules of the house style:
//    • transparent background (sits on the page's paper), Inter everywhere
//    • horizontal gridlines only (gridY); no chart border, no vertical grid
//    • one primary hue leads; a second hue (never grey) carries the comparison
//    • every data mark is interactive: pass tip:true so hovering shows values
// ─────────────────────────────────────────────────────────────────────────

export const C = {
  paper: "#fffcf0", // flexoki paper  — page background
  ink: "#100f0f", // flexoki black  — text, axis labels
  grid: "#e6e4d9", // flexoki-100    — gridlines
  faint: "#b7b5ac", // flexoki-300    — reference lines, footnotes
  muted: "#6f6e69", // flexoki-600    — secondary labels

  primary: "#ce5d97", // flexoki magenta-400 (pink)   — series A / "self" / hero
  secondary: "#3aa99f", // flexoki cyan-400 (bright teal) — series B / "other"
  accent: "#a02f6f", // flexoki magenta-600 (deep)   — tertiary / steered (a
  //                    darker shade of self, reads as "self, intervened")
  neutral: "#c3c1b6", // flexoki-200-ish              — muted control / placebo
};

// Categorical order for multi-series figures (e.g. the steering conditions).
export const CATEGORICAL = [C.primary, C.accent, C.neutral, C.secondary];

export const FONT =
  "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";

// Common Plot top-level style. Use as:  Plot.plot({ style: plotStyle(), ... })
export function plotStyle(extra = {}) {
  return {
    background: "transparent",
    color: C.ink,
    fontFamily: FONT,
    fontSize: "13px",
    ...extra,
  };
}

// Consistent gridlines (call inside marks: [ gridY(Plot), ... ]).
export const gridY = (Plot) => Plot.gridY({ stroke: C.grid, strokeOpacity: 1 });
export const gridX = (Plot) => Plot.gridX({ stroke: C.grid, strokeOpacity: 1 });
