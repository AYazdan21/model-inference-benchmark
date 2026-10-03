"""
Self-contained HTML rendering of a benchmark report: inline CSS, charts as inline SVG generated here,
sample frames embedded as base64 JPEG (or linked as samples/*.jpg inside a benchmark suite). No scripts, CDNs or external files, so the file can be emailed,
archived or printed to PDF from any browser.
"""
import base64
import math
from html import escape
from typing import Any, Dict, List, Optional, Sequence, Tuple

CSS = """
:root {
  --bg:#f4f6fa; --card:#ffffff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --grid:#e2e8f0;
  --host:#2563eb; --dev:#ea580c; --pre:#0d9488; --post:#7c3aed; --dec:#94a3b8;
  --ok:#15803d; --warn:#b45309; --fail:#b91c1c; --ok-bg:#dcfce7; --warn-bg:#fef3c7; --fail-bg:#fee2e2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0b1220; --card:#131c2e; --text:#e5e7eb; --muted:#94a3b8; --border:#263248; --grid:#263248;
    --host:#60a5fa; --dev:#fb923c; --pre:#2dd4bf; --post:#a78bfa; --dec:#64748b;
    --ok:#4ade80; --warn:#fbbf24; --fail:#f87171; --ok-bg:#14361f; --warn-bg:#3b2f0d; --fail-bg:#3f1616;
  }
}
html { color-scheme: light dark; }
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--text); line-height:1.5;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 4px; }
h2 { font-size:1.1rem; margin:0 0 10px; }
.sub { color:var(--muted); font-size:.9rem; }
.head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; margin-bottom:18px; }
.badge { display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; letter-spacing:.04em; font-size:.85rem; }
.badge.ok { background:var(--ok-bg); color:var(--ok); }
.badge.warn { background:var(--warn-bg); color:var(--warn); }
.badge.fail { background:var(--fail-bg); color:var(--fail); }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px 18px; margin-bottom:16px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; margin-bottom:16px; }
.kpi { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
.kpi .v { font-size:1.4rem; font-weight:700; }
.kpi .v small { font-size:.8rem; font-weight:500; color:var(--muted); }
.kpi .l { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.kpi .n { font-size:.75rem; color:var(--muted); }
.kpi.ok .v { color:var(--ok); } .kpi.warn .v { color:var(--warn); } .kpi.fail .v { color:var(--fail); }
ul.verdict { list-style:none; margin:0; padding:0; }
ul.verdict li { padding:6px 0; border-bottom:1px solid var(--border); display:flex; gap:10px; align-items:flex-start; }
ul.verdict li:last-child { border-bottom:none; }
.pill { flex:none; min-width:48px; text-align:center; font-size:.7rem; font-weight:700; padding:2px 8px; border-radius:6px; }
.pill.ok { background:var(--ok-bg); color:var(--ok); } .pill.warn { background:var(--warn-bg); color:var(--warn); }
.pill.fail { background:var(--fail-bg); color:var(--fail); }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }
.charts .card { margin-bottom:0; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:.78rem; color:var(--muted); margin-bottom:4px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:-1px; }
.note { font-size:.78rem; color:var(--muted); margin:4px 0 0; }
svg { width:100%; height:auto; display:block; }
svg text { fill:var(--muted); font-size:12.5px; font-family:inherit; }
svg .grid { stroke:var(--grid); stroke-width:1; }
svg .axis { stroke:var(--muted); stroke-width:1; }
svg .line-host { stroke:var(--host); fill:none; stroke-width:1.8; } svg .dot-host { fill:var(--host); }
svg .line-dev { stroke:var(--dev); fill:none; stroke-width:1.8; } svg .dot-dev { fill:var(--dev); }
svg .budget { stroke:var(--fail); stroke-dasharray:5 4; stroke-width:1.2; }
svg .fill-host { fill:var(--host); } svg .fill-dev { fill:var(--dev); } svg .fill-pre { fill:var(--pre); }
svg .fill-post { fill:var(--post); } svg .fill-dec { fill:var(--dec); }
svg .seglbl { fill:#fff; font-size:11px; }
svg .empty { font-size:13px; }
table { width:100%; border-collapse:collapse; font-size:.85rem; }
th, td { padding:5px 8px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase; letter-spacing:.03em; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
table.kv td:first-child { color:var(--muted); width:38%; }
td:first-child { white-space:nowrap; }
table.kv td:first-child { white-space:normal; }
.tables { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:16px; }
.tables .card { margin-bottom:0; }
.samples { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
.samples img { width:100%; border-radius:8px; border:1px solid var(--border); }
.samples figcaption { font-size:.78rem; color:var(--muted); }
figure { margin:0; }
.back { margin:0 0 10px; font-size:.85rem; } .back a { color:var(--host); text-decoration:none; } .back a:hover { text-decoration:underline; }
.notes { white-space:pre-wrap; overflow-wrap:anywhere; unicode-bidi:plaintext; }
td.notes { font-size:.8rem; max-width:340px; min-width:120px; }
.notes-card table.kv td:first-child { width:90px; font-weight:600; }
.notes-card td.notes { max-width:none; font-size:.9rem; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.8rem; word-break:break-all; }
@media (max-width:560px) { body { padding:12px; } .charts, .tables { grid-template-columns:1fr; } }
@media print {
  :root { --bg:#fff; --card:#fff; --text:#000; --muted:#475569; --border:#cbd5e1; --grid:#e2e8f0;
    --host:#2563eb; --dev:#ea580c; --pre:#0d9488; --post:#7c3aed; --dec:#94a3b8;
    --ok:#15803d; --warn:#b45309; --fail:#b91c1c; --ok-bg:#dcfce7; --warn-bg:#fef3c7; --fail-bg:#fee2e2; }
  body { padding:0; } .wrap { max-width:none; }
  .card, .kpi, figure { break-inside:avoid; }
  .charts, .tables { grid-template-columns:1fr 1fr; }
  * { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
"""

CSS += """
:root { --s1:#2563eb; --s2:#ea580c; --s3:#0d9488; --s4:#7c3aed; --s5:#ca8a04; --s6:#db2777; }
@media (prefers-color-scheme: dark) { :root { --s1:#60a5fa; --s2:#fb923c; --s3:#2dd4bf; --s4:#a78bfa; --s5:#facc15; --s6:#f472b6; } }
@media print { :root { --s1:#2563eb; --s2:#ea580c; --s3:#0d9488; --s4:#7c3aed; --s5:#ca8a04; --s6:#db2777; } }
svg .line-s1 { stroke:var(--s1); fill:none; stroke-width:1.8; } svg .dot-s1 { fill:var(--s1); }
svg .line-s2 { stroke:var(--s2); fill:none; stroke-width:1.8; } svg .dot-s2 { fill:var(--s2); }
svg .line-s3 { stroke:var(--s3); fill:none; stroke-width:1.8; } svg .dot-s3 { fill:var(--s3); }
svg .line-s4 { stroke:var(--s4); fill:none; stroke-width:1.8; } svg .dot-s4 { fill:var(--s4); }
svg .line-s5 { stroke:var(--s5); fill:none; stroke-width:1.8; } svg .dot-s5 { fill:var(--s5); }
svg .line-s6 { stroke:var(--s6); fill:none; stroke-width:1.8; } svg .dot-s6 { fill:var(--s6); }
.best-banner { border-left:5px solid var(--ok); }
.best-banner .l { font-size:.72rem; color:var(--muted); letter-spacing:.06em; font-weight:700; }
.best-banner .big { font-size:1.25rem; font-weight:700; margin:2px 0 6px; }
.best-banner > div { margin:2px 0; }
table.cfgtable { font-size:.8rem; }
table.cfgtable td { white-space:nowrap; }
table.cfgtable tr.best td { background:var(--ok-bg); font-weight:600; }
td.rt-yes { color:var(--ok); font-weight:700; } td.rt-no { color:var(--fail); font-weight:700; }
.tag { font-size:.62rem; border:1px solid var(--border); border-radius:5px; padding:0 4px; color:var(--muted); margin-left:4px; }
details.cfgsamples { border-top:1px solid var(--border); padding:6px 0; }
details.cfgsamples summary { cursor:pointer; font-size:.85rem; }
details.cfgsamples .samples { margin-top:8px; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }
td.s, .s { color:var(--muted); }
"""


# ---------------------------------------------------------------- formatting helpers

def _e(v: Any) -> str:
    return escape("" if v is None else str(v))


def _f(v: Any, nd: int = 2, unit: str = "") -> str:
    if v is None or v == "":
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        txt = str(v)
        if isinstance(v, float):
            txt = f"{v:.{nd}f}"
            if "." in txt:
                txt = txt.rstrip("0").rstrip(".")
        return f"{txt}{unit}"
    return _e(v)


def _num(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp


# ---------------------------------------------------------------- SVG charts

W, H = 640, 260
ML, MR, MT, MB = 56, 14, 12, 46


def _tick_count(ymax: float) -> int:
    """Number of y intervals whose step is a round number (e.g. 250 -> 5 x 50, not 4 x 62.5)."""
    for n in (4, 5, 2, 3):
        step = f"{ymax / n:.6g}"
        if len(step.replace(".", "").strip("0")) <= 2:
            return n
    return 4


def _frame(xlabel: str, ylabel: str, ymax: float, yticks: Optional[int] = None) -> Tuple[List[str], float, float, float, float]:
    """Axes, y grid and axis titles; returns (parts, plot_x0, plot_x1, plot_y0(top), plot_y1(bottom))."""
    x0, x1, y0, y1 = ML, W - MR, MT, H - MB
    yticks = yticks or _tick_count(ymax)
    parts = []
    for i in range(yticks + 1):
        val = ymax * i / yticks
        y = y1 - (y1 - y0) * i / yticks
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{_num(val)}</text>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    parts.append(f'<text x="{(x0 + x1) / 2}" y="{H - 6}" text-anchor="middle">{_e(xlabel)}</text>')
    parts.append(f'<text transform="translate(13 {(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">{_e(ylabel)}</text>')
    return parts, x0, x1, y0, y1


def _svg(parts: Sequence[str], label: str) -> str:
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_e(label)}">' + "".join(parts) + "</svg>"


def _xticks(parts: List[str], xs_min: float, xs_max: float, x0: float, x1: float, y1: float, count: int = 6) -> None:
    span = xs_max - xs_min
    n = min(count, max(1, int(span) + 1))
    for i in range(n + 1 if n > 1 else 1):
        v = xs_min + span * i / n if n > 1 else xs_min
        x = x0 + (x1 - x0) * ((v - xs_min) / span if span else 0.5)
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 4}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" text-anchor="middle">{v:.0f}</text>')


def line_chart(series: List[Tuple[str, List[float], List[float]]], xlabel: str, ylabel: str,
               budget: Optional[float] = None, title: str = "", ymax: Optional[float] = None) -> str:
    """series: [(css suffix 'host'|'dev'|'s1'..'s6', xs, ys)]. ymax fixes the top of the y axis."""
    all_y = [y for _, _, ys in series for y in ys]
    all_x = [x for _, xs, _ in series for x in xs]
    if not all_y:
        return _svg([f'<text class="empty" x="{W / 2}" y="{H / 2}" text-anchor="middle">No data</text>'], title)
    data_max = max(all_y)
    show_budget = budget is not None and budget <= 3 * data_max
    top = ymax or _nice_max(max(data_max, budget if show_budget else 0) * 1.05)
    parts, x0, x1, y0, y1 = _frame(xlabel, ylabel, top)
    xmin, xmax = min(all_x), max(all_x)
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    px = lambda x: x0 + (x1 - x0) * (x - xmin) / (xmax - xmin)
    py = lambda y: y1 - (y1 - y0) * y / top
    _xticks(parts, xmin if xmin % 1 == 0 else math.ceil(xmin), xmax if xmax % 1 == 0 else math.floor(xmax), x0, x1, y1)
    if show_budget:
        parts.append(f'<line class="budget" x1="{x0}" y1="{py(budget):.1f}" x2="{x1}" y2="{py(budget):.1f}"/>')
    for css, xs, ys in series:
        if len(xs) == 1:
            parts.append(f'<circle class="dot-{css}" cx="{px(xs[0]):.1f}" cy="{py(ys[0]):.1f}" r="4"/>')
        else:
            pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
            parts.append(f'<polyline class="line-{css}" points="{pts}"/>')
    return _svg(parts, title)


def bar_chart(values: List[float], xs: List[float], xlabel: str, ylabel: str, css: str, title: str,
              empty_text: Optional[str] = None) -> str:
    if not values:
        return _svg([f'<text class="empty" x="{W / 2}" y="{H / 2}" text-anchor="middle">No data</text>'], title)
    top = _nice_max(max(max(values), 1))
    parts, x0, x1, y0, y1 = _frame(xlabel, ylabel, top)
    n = len(values)
    slot = (x1 - x0) / n
    bw = max(1.0, slot * 0.78)
    for i, v in enumerate(values):
        if v <= 0:
            continue
        h = (y1 - y0) * v / top
        parts.append(f'<rect class="fill-{css}" x="{x0 + slot * i + (slot - bw) / 2:.1f}" y="{y1 - h:.1f}" '
                     f'width="{bw:.1f}" height="{h:.1f}"/>')
    step = max(1, math.ceil(n / 8))
    for i in range(0, n, step):
        x = x0 + slot * (i + 0.5)
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" text-anchor="middle">{xs[i]:.0f}</text>')
    if empty_text and max(values) <= 0:
        parts.append(f'<text class="empty" x="{(x0 + x1) / 2}" y="{(y0 + y1) / 2}" text-anchor="middle">{_e(empty_text)}</text>')
    return _svg(parts, title)


def histogram(series: List[Tuple[str, List[float]]], xlabel: str, title: str, bins: int = 12) -> str:
    vals = [v for _, vs in series for v in vs]
    if not vals:
        return _svg([f'<text class="empty" x="{W / 2}" y="{H / 2}" text-anchor="middle">No data</text>'], title)
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / bins
    counts = []
    for css, vs in series:
        c = [0] * bins
        for v in vs:
            c[min(bins - 1, int((v - lo) / width))] += 1
        counts.append((css, c))
    top = _nice_max(max(max(c) for _, c in counts))
    parts, x0, x1, y0, y1 = _frame(xlabel, "Frames", top)
    slot = (x1 - x0) / bins
    for css, c in counts:
        for i, n in enumerate(c):
            if n:
                h = (y1 - y0) * n / top
                parts.append(f'<rect class="fill-{css}" fill-opacity="0.6" x="{x0 + slot * i + 1:.1f}" y="{y1 - h:.1f}" '
                             f'width="{slot - 2:.1f}" height="{h:.1f}"/>')
    for i in range(0, bins + 1, max(1, bins // 6)):
        x = x0 + slot * i
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" text-anchor="middle">{_num(lo + width * i)}</text>')
    return _svg(parts, title)


def stacked_bars(rows: List[Tuple[str, List[Tuple[str, float, str]]]], title: str,
                 label_w: int = 120, label_chars: Optional[int] = None) -> str:
    """rows: [(label, [(segment name, ms, css)])] drawn as horizontal stacked bars on a shared ms axis."""
    totals = [sum(v for _, v, _ in segs) for _, segs in rows]
    top = _nice_max(max(totals) * 1.02) if totals and max(totals) > 0 else 1.0
    x0, x1 = label_w, W - 20
    bar_h, gap, y = 34, 26, 16
    parts = []
    n_ticks = _tick_count(top)
    for tick in range(n_ticks + 1):
        x = x0 + (x1 - x0) * tick / n_ticks
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{y + len(rows) * (bar_h + gap) - gap + 8}"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + len(rows) * (bar_h + gap) + 8}" text-anchor="middle">{_num(top * tick / n_ticks)}</text>')
    axis_y = y + len(rows) * (bar_h + gap) + 26
    parts.append(f'<text x="{(x0 + x1) / 2}" y="{axis_y}" text-anchor="middle">Milliseconds per frame</text>')
    for label, segs in rows:
        shown = label if label_chars is None or len(label) <= label_chars else label[:label_chars - 1] + "…"
        parts.append(f'<text x="{x0 - 8}" y="{y + bar_h / 2 + 4}" text-anchor="end"><title>{_e(label)}</title>{_e(shown)}</text>')
        cx = x0
        for name, val, css in segs:
            w = (x1 - x0) * val / top
            if w <= 0:
                continue
            parts.append(f'<rect class="fill-{css}" x="{cx:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}"><title>{_e(name)}: {val:.2f} ms</title></rect>')
            if w > 44:
                parts.append(f'<text class="seglbl" x="{cx + w / 2:.1f}" y="{y + bar_h / 2 + 4}" text-anchor="middle">{val:.1f}</text>')
            cx += w
        y += bar_h + gap
    return f'<svg viewBox="0 0 {W} {axis_y + 10:.0f}" role="img" aria-label="{_e(title)}">' + "".join(parts) + "</svg>"


def hbars(items: List[Tuple[Any, ...]], xlabel: str, title: str, css: str = "host", limit: int = 12,
          label_chars: int = 20, label_w: int = 130, marker: Optional[float] = None, marker_label: str = "") -> str:
    """Horizontal bars. items: (name, value) or (name, value, css) for a per-bar colour.
    marker draws a dashed vertical line (e.g. a required FPS or a RAM budget) on the value axis."""
    if not items:
        return _svg([f'<text class="empty" x="{W / 2}" y="{H / 2}" text-anchor="middle">No detections</text>'], title)
    items = items[:limit]
    top = _nice_max(max(max(it[1] for it in items), (marker or 0) * 1.05))
    x0, x1 = label_w, W - 40
    row = 24
    height = row * len(items) + 44 + (16 if marker and marker_label else 0)
    parts = []
    n_ticks = _tick_count(top)
    for tick in range(n_ticks + 1):
        x = x0 + (x1 - x0) * tick / n_ticks
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{row * len(items) + 4}"/>')
        parts.append(f'<text x="{x:.1f}" y="{row * len(items) + 18}" text-anchor="middle">{_num(top * tick / n_ticks)}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2}" y="{height - 4}" text-anchor="middle">{_e(xlabel)}</text>')
    for i, it in enumerate(items):
        name, v = it[0], it[1]
        bar_css = it[2] if len(it) > 2 else css
        y = 4 + i * row
        w = (x1 - x0) * v / top
        shown = name if len(name) <= label_chars else name[:label_chars - 1] + "…"
        parts.append(f'<text x="{x0 - 8}" y="{y + 15}" text-anchor="end"><title>{_e(name)}</title>{_e(shown)}</text>')
        parts.append(f'<rect class="fill-{bar_css}" x="{x0}" y="{y + 3}" width="{max(w, 0):.1f}" height="{row - 8}"><title>{_e(name)}: {_num(v)}</title></rect>')
        parts.append(f'<text x="{x0 + w + 5:.1f}" y="{y + 15}">{_num(v)}</text>')
    if marker:
        mx = x0 + (x1 - x0) * marker / top
        parts.append(f'<line class="budget" x1="{mx:.1f}" y1="0" x2="{mx:.1f}" y2="{row * len(items) + 4}"/>')
        if marker_label:
            parts.append(f'<text x="{mx:.1f}" y="{row * len(items) + 32}" text-anchor="middle">{_e(marker_label)}</text>')
    return f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{_e(title)}">' + "".join(parts) + "</svg>"


# ---------------------------------------------------------------- page assembly

def _legend(items: Sequence[Tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:var(--{c})"></i>{_e(t)}</span>' for t, c in items) + "</div>"


def _chart_card(title: str, legend: str, svg: str, note: str = "") -> str:
    return f'<div class="card"><h2>{_e(title)}</h2>{legend}{svg}' + (f'<p class="note">{_e(note)}</p>' if note else "") + "</div>"


def _kv(title: str, rows: List[Tuple[str, Any]]) -> str:
    body = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v if isinstance(v, str) else _f(v))}</td></tr>" for k, v in rows)
    return f'<div class="card"><h2>{_e(title)}</h2><table class="kv">{body}</table></div>'


def _kpi(label: str, value: str, unit: str = "", note: str = "", level: str = "") -> str:
    return (f'<div class="kpi {level}"><div class="l">{_e(label)}</div>'
            f'<div class="v">{_e(value)}{f" <small>{_e(unit)}</small>" if unit else ""}</div>'
            f'<div class="n">{_e(note)}</div></div>')


def render_html(report: Dict[str, Any], per_frame: List[Dict[str, Any]], samples: List[Tuple[int, bytes]],
                link_samples: bool = False, back_link: Optional[Tuple[str, str]] = None) -> str:
    """link_samples: reference samples/frame_XXXXXX.jpg next to the report instead of embedding base64 (suite runs).
    back_link: (href, label) shown at the top, used when the report lives inside a benchmark suite."""
    t, m, cfg, cold = report["target"], report["model"], report["config"], report["cold_start"]
    host, est, rt, res = report["host_performance"], report["device_estimate"], report["realtime"], report["resources"]
    dets, verdict, env = report["detections"], report["verdict"], report["environment"]
    src = cfg["source"]
    overall = verdict["overall"]
    simulated = est is not None
    lat = host["latency_ms"]
    sweep = report.get("sweep")
    if sweep:
        from src.benchmark import sweep_html

    # ---- KPI tiles
    rt_level = "ok" if rt["realtime_capable"] else ("warn" if rt["device_realtime_factor"] >= 0.5 else "fail")
    ram_level = "fail" if res["ram_limit_breached"] else ""
    kpis = []
    if simulated:
        kpis += [
            _kpi("Est. device FPS", _f(est["est_fps"]), "FPS", f"{rt['device_realtime_factor']:.2f}x of {rt['required_fps']:g} FPS needed", rt_level),
            _kpi("Est. latency P50 / P95", f"{_f(est['latency_ms']['p50'], 1)} / {_f(est['latency_ms']['p95'], 1)}", "ms", "on " + t["device"]),
        ]
    else:
        kpis += [_kpi("Host FPS (measured)", _f(host["throughput_fps"]), "FPS", f"{rt['device_realtime_factor']:.2f}x of {rt['required_fps']:g} FPS needed", rt_level)]
    kpis += [
        _kpi("Host FPS" if simulated else "Host latency P50 / P95",
             _f(host["throughput_fps"]) if simulated else f"{_f(lat['p50'], 1)} / {_f(lat['p95'], 1)}",
             "FPS" if simulated else "ms", "measured, processing only" if simulated else "measured on this machine"),
        _kpi("Detections / frame", _f(dets["mean_per_frame"]), "", f"{dets['total']} total, {dets['frames_with_detections_pct']}% of frames"),
        _kpi("Peak RAM", _f(res["peak_ram_mb"], 0), "MB", f"limit {_f(res['ram_limit_mb'], 0)} MB" if res["ram_limit_mb"] else "no limit", ram_level),
        _kpi("Avg CPU load", _f(res["avg_cpu_percent"], 0), "%", f"of {res['cpu_cores'] or 'all'} target cores"),
    ]

    # ---- verdict
    verdict_html = "".join(f'<li><span class="pill {i["level"]}">{i["level"].upper()}</span><span>{_e(i["message"])}</span></li>' for i in verdict["items"])

    # ---- charts
    idx = [r["frame_index"] for r in per_frame]
    host_ms = [r["host_latency_ms"] for r in per_frame]
    dev_pairs = [(r["frame_index"], r["sim_latency_ms"]) for r in per_frame if r["sim_latency_ms"] is not None]
    series = [("host", idx, host_ms)]
    legend_items = [("Host (measured)", "host")]
    if simulated:
        series.append(("dev", [i for i, _ in dev_pairs], [v for _, v in dev_pairs]))
        legend_items.append((f"{t['device']} (estimated)", "dev"))
    budget = rt["frame_budget_ms"]
    line_svg = line_chart(series, "Frame index", "Latency per frame (ms)", budget, "Latency per frame")
    line_note = f"Dashed red line: {budget} ms frame budget for {rt['required_fps']:g} FPS." if budget and budget <= 3 * max(host_ms + [v for _, v in dev_pairs] or [0]) else ""

    hist_series = [("host", host_ms)] + ([("dev", [v for _, v in dev_pairs])] if simulated else [])
    hist_svg = histogram(hist_series, "Latency per frame (ms)", "Latency distribution")

    def _mean(key: str) -> float:
        vals = [r[key] for r in per_frame if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    host_segments = []
    if host["stages"].get("decode"):
        host_segments.append(("Decode", _mean("decode_ms"), "dec"))
    host_segments += [("Preprocess", _mean("preprocess_ms"), "pre"), ("Inference", _mean("inference_ms"), "host"), ("Postprocess", _mean("postprocess_ms"), "post")]
    stack_rows = [("Host (measured)", host_segments)]
    if simulated:
        stack_rows.append(("Device (est.)", [("CPU-side (pre+post)", est["cpu_side_ms"] or 0.0, "pre"), ("Inference", est["inference_ms"] or 0.0, "dev")]))
    stack_legend = [("Decode", "dec"), ("Preprocess", "pre"), ("Inference (host)", "host"), ("Postprocess", "post")]
    if simulated:
        stack_legend += [("Inference (device est.)", "dev")]
    stack_svg = stacked_bars(stack_rows, "Stage breakdown")

    det_svg = bar_chart([float(r["detections"]) for r in per_frame], [float(i) for i in idx], "Frame index", "Detections",
                        "host", "Detections per frame", "No detections in any frame")
    class_items = [(f"class {name}" if name.isdigit() else name, float(v["count"])) for name, v in dets["per_class"].items()]
    class_svg = hbars(class_items, "Detections", "Detections per class")

    charts = "".join([
        _chart_card("Latency per frame", _legend(legend_items), line_svg, line_note),
        _chart_card("Latency distribution", _legend(legend_items), hist_svg),
        _chart_card("Stage breakdown (mean per frame)", _legend(stack_legend), stack_svg,
                    "Decode is timed separately and excluded from latency and FPS." + ("" if simulated else " Host measurement only: this target is not simulated.")),
        _chart_card("Detections per frame", "", det_svg, "Measured on the host with the real model."),
        _chart_card("Detections per class", "", class_svg),
    ])

    # ---- detail tables
    def lat_row(label: str, s: Optional[Dict[str, Any]]) -> str:
        if not s:
            return ""
        cells = "".join(f'<td class="num">{_f(s.get(k))}</td>' for k in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"))
        return f"<tr><td>{_e(label)}</td>{cells}</tr>"

    lat_rows = lat_row("Host total (ms)", lat)
    if simulated:
        lat_rows += lat_row("Device est. total (ms)", est["latency_ms"])
    lat_table = ('<div class="card"><h2>Latency statistics</h2><div style="overflow-x:auto"><table><tr><th></th>'
                 + "".join(f'<th class="num">{h}</th>' for h in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"))
                 + f"</tr>{lat_rows}</table></div>")
    stage_rows = "".join(f'<tr><td>{n}</td><td class="num">{_f((host["stages"].get(k) or {}).get("mean"))}</td>'
                         f'<td class="num">{_f((host["stages"].get(k) or {}).get("p95"))}</td></tr>'
                         for n, k in (("Decode (not in total)", "decode"), ("Preprocess", "preprocess"), ("Inference", "inference"), ("Postprocess", "postprocess")))
    lat_table += f'<table style="margin-top:12px"><tr><th>Host stage (ms)</th><th class="num">mean</th><th class="num">p95</th></tr>{stage_rows}</table></div>'

    cls_rows = "".join(f'<tr><td>{_e(n)}</td><td class="num">{v["count"]}</td><td class="num">{_f(v["mean_conf"], 3)}</td><td class="num">{_f(v["max_conf"], 3)}</td></tr>'
                       for n, v in dets["per_class"].items())
    det_table = _kv("Detections", [
        ("Total", dets["total"]), ("Mean / max per frame", f"{_f(dets['mean_per_frame'], 3)} / {dets['max_per_frame']}"),
        ("Frames with detections", f"{dets['frames_with_detections']} ({dets['frames_with_detections_pct']}%)"),
        ("Confidence mean / min / max", f"{_f(dets['confidence']['mean'], 3)} / {_f(dets['confidence']['min'], 3)} / {_f(dets['confidence']['max'], 3)}"),
        ("Output format", m["output_format"]),
    ])
    if cls_rows:
        det_table = det_table.replace("</table></div>", f'</table><table style="margin-top:12px"><tr><th>Class</th><th class="num">count</th><th class="num">mean conf</th><th class="num">max conf</th></tr>{cls_rows}</table></div>')

    cal = t.get("calibration")
    tables = [
        lat_table,
        _kv("Real-time", [
            ("Required FPS", rt["required_fps"]), ("Basis", rt["basis"]), ("Real-time factor", f"{rt['device_realtime_factor']}x"),
            ("Real-time capable", rt["realtime_capable"]), ("Frame budget", f"{_f(rt['frame_budget_ms'])} ms"),
            ("P95 within budget", rt["p95_within_budget"]),
            ("Frames analysed", "all frames" if rt["analyse_every_n_frames"] == 1 else f"1 of every {rt['analyse_every_n_frames']} frames"),
        ]),
        _kv("Cold start", [
            ("Model load", f"{_f(cold['model_load_ms'], 1)} ms"), ("First inference", f"{_f(cold['first_inference_ms'], 1)} ms"),
            ("RSS before load", f"{_f(cold['rss_before_load_mb'], 1)} MB"), ("RSS after load", f"{_f(cold['rss_after_load_mb'], 1)} MB"),
            ("RSS after warmup", f"{_f(cold.get('rss_after_warmup_mb'), 1)} MB"),
        ]),
        _kv("Resources", [
            ("RAM limit", f"{_f(res['ram_limit_mb'], 0)} MB" if res["ram_limit_mb"] else "none"), ("Peak RAM", f"{_f(res['peak_ram_mb'], 1)} MB"),
            ("RAM headroom", f"{_f(res['ram_headroom_mb'], 1)} MB" if res["ram_headroom_mb"] is not None else "n/a"),
            ("RAM limit breached", res["ram_limit_breached"]), ("CPU avg / peak (of target cores)", f"{_f(res['avg_cpu_percent'], 1)} / {_f(res['peak_cpu_percent'], 1)} %"),
            ("VRAM limit / peak", f"{_f(res['vram_limit_mb'], 0)} / {_f(res['peak_vram_mb'], 1)} MB" if res["vram_limit_mb"] else "n/a"),
        ]),
        det_table,
        _kv("Device estimate", [
            ("Est. inference", f"{_f(est['inference_ms'])} ms"), ("Est. CPU-side (pre+post)", f"{_f(est['cpu_side_ms'])} ms"),
            ("Est. FPS", est["est_fps"]), ("CPU scale (host to device)", t["cpu_scale"]),
            ("Calibration anchor", f"{cal['model']} = {cal['latency_ms']} ms ({cal['gflops']} GFLOPs)" if cal else "n/a"),
            ("Anchor source", cal["source"] if cal else "n/a"),
        ] if simulated else [("Status", "Not simulated: host measurements only")]),
        _kv("Target", [
            ("Key", t["key"]), ("Device", t["device"]), ("Description", t["description"]), ("Compute unit", t["compute_unit"]),
            ("Runtime modelled", t["runtime"]), ("CPU cores", t["cpu_cores"] or "all"),
            ("RAM / VRAM budget", f"{_f(t['ram_limit_mb'], 0)} / {_f(t['vram_limit_mb'], 0)} MB"),
        ]),
        _kv("Model", [
            ("File", m["file"]), ("Format", m["format"]), ("Size", f"{m['size_mb']} MB"), ("SHA-256 (first 16)", m["sha256"]),
            ("Compute", f"{_f(m['gflops'], 3)} GFLOPs, {_f(m['params_m'], 3)} M params"),
            ("Input size (h, w)", " x ".join(str(x) for x in m["input_size"]) if m["input_size"] else "n/a"),
            ("Classes", ", ".join(m["class_names"]) if m["class_names"] else "not stored in model"),
            ("Output format", m["output_format"]),
        ]),
        _kv("Run configuration", [
            ("Source", src.get("file") or "synthetic random frames"), ("Resolution", f"{src.get('width')} x {src.get('height')}"),
            ("Source FPS", src.get("fps")), ("Frames processed / requested", f"{cfg['frames_processed']} / {cfg['frames_requested']}"),
            ("Warmup iterations", cfg["warmup"]), ("Confidence / IoU threshold", f"{cfg['conf_threshold']} / {cfg['iou_threshold']}"),
            ("Wall time", f"{host['elapsed_seconds']} s (wall FPS {host['wall_fps']})"),
        ] + ([("Source resolution used (best configuration)", cfg["source_resolution"]["text"]),
              ("Model input size used (best configuration)", " x ".join(str(x) for x in cfg["input_size"]))] if sweep else [])),
        _kv("Environment", [
            ("Execution", env["execution"]), ("Host name", env["hostname"]), ("Platform", env["platform"]),
            ("CPU", env["cpu_model"]), ("Logical CPUs visible", env["logical_cpus"]),
            ("cgroup CPU / memory limit", f"{_f(env['cgroup']['cpu_limit_cores'])} cores / {_f(env['cgroup']['memory_limit_mb'], 0)} MB"),
            ("Python", env["python"]),
            ("Packages", ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v)),
        ]),
    ]

    # ---- sample frames
    sample_html = ""
    if samples:
        rel_of = {int(r.rsplit("_f", 1)[-1][:-4]): r for r in report.get("samples") or [] if "_f" in r} if sweep else {}
        link = (lambda i, j: rel_of.get(i) or f"samples/frame_{i:06d}.jpg") if sweep else (lambda i, j: f"samples/frame_{i:06d}.jpg")
        src_of = link if link_samples else (lambda i, j: "data:image/jpeg;base64," + base64.b64encode(j).decode("ascii"))
        figs = "".join(
            f'<figure><img alt="Annotated frame {i}" src="{src_of(i, j)}" loading="lazy">'
            f"<figcaption>Frame {i}</figcaption></figure>" for i, j in samples)
        sample_html = f'<div class="card"><h2>Sample frames{" of the best configuration" if sweep else " (most detections)"}</h2><div class="samples">{figs}</div></div>'

    title = f"{m['file']} on {t['device']}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark report: {_e(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  {f'<p class="back"><a href="{_e(back_link[0])}">&larr; {_e(back_link[1])}</a></p>' if back_link else ''}
  <div class="head">
    <div>
      <h1>{_e(m['file'])} <span class="sub">on</span> {_e(t['device'])}</h1>
      <div class="sub">{_e(t['key'])} &middot; {_e(t['runtime'])} &middot; {_e(report['created'].replace('T', ' '))} &middot; <span class="mono">{_e(report['report_id'])}</span></div>
    </div>
    <span class="badge {overall}">{overall.upper()}</span>
  </div>
  <div class="card notes-card"><table class="kv"><tr><td>Notes</td><td class="notes" dir="auto">{_e(report.get('notes', ''))}</td></tr></table></div>
  {sweep_html.best_banner(report) if sweep else ''}
  <div class="kpis">{''.join(kpis)}</div>
  <div class="card"><h2>Verdict</h2><ul class="verdict">{verdict_html}</ul></div>
  {sweep_html.config_table(report) + sweep_html.sweep_charts(report) + sweep_html.sample_grid(report) if sweep else ''}
  {'<h2 style="margin:22px 0 10px">Best configuration: detail charts</h2>' if sweep else ''}
  <div class="charts">{charts}</div>
  <h2 style="margin:22px 0 10px">{'Details of the best configuration' if sweep else 'Details'}</h2>
  <div class="tables">{''.join(tables)}</div>
  {sample_html}
  {sweep_html.rules_card() if sweep else ''}
  <div class="card" style="margin-top:16px"><h2>Method and accuracy</h2>
    <p>{_e(t['method'])}</p><p class="note">{_e(t['accuracy_disclaimer'])}</p>
    <p class="note">Schema version {_e(report['schema_version'])}. Generated by Crime-Detect. Use the browser's Print to save this page as PDF.</p>
  </div>
</div>
</body>
</html>
"""
