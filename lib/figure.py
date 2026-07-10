"""The figure -> blog pipeline.

A figure is DATA (data.json) + a RENDER function (plot.js, Observable Plot), kept separate so the
data can change without touching the render code.

    from lib import figure
    fig = figure.save(PROJECT_DIR, "datacenter-buildout", data, plot_js)
    figure.publish(fig)   # -> blog.kran.ai/assets/figures/datacenter-buildout/

Then embed in a blog post (see CLAUDE.md):
    {% include figure.html id="datacenter-buildout" caption="..." %}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

BLOG_FIGURES = Path.home() / "Desktop" / "Kran" / "blog.kran.ai" / "assets" / "figures"


def save(project_dir, fig_id: str, data, plot_js: str | None = None) -> Path:
    """Write figures/<fig_id>/{data.json, plot.js} inside the project. Returns the figure dir."""
    out = Path(project_dir) / "figures" / fig_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=2))
    if plot_js is not None:
        (out / "plot.js").write_text(plot_js)
    return out


def publish(fig_dir, blog_figures: Path = BLOG_FIGURES) -> Path:
    """Copy a figure dir into the blog's assets/figures/. Returns the destination dir.

    Does not touch the blog's git — commit/deploy is a human step (see CLAUDE.md).
    """
    fig_dir = Path(fig_dir)
    dest = Path(blog_figures) / fig_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("data.json", "plot.js"):
        src = fig_dir / f
        if src.exists():
            shutil.copy2(src, dest / f)
    return dest


def publish_theme(theme_js, blog_figures: Path = BLOG_FIGURES) -> Path:
    """Copy the shared figure style guide to blog assets/figures/theme.js.

    Every plot.js imports it as `../theme.js`, so it must live one level above the
    per-figure dirs. Run this whenever theme.js changes.
    """
    dest = Path(blog_figures) / "theme.js"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(theme_js), dest)
    return dest
