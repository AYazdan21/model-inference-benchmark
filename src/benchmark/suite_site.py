"""
Multipage, self-contained HTML site for a benchmark suite (see suite.py for the folder layout).

Pages (inline CSS, inline SVG, no scripts, relative links only, so the folder can be zipped, e-mailed or opened from disk):
    index.html            overview: KPIs, FPS heatmap (model x device), best model per device, real-time matrix,
                          RAM-vs-budget heatmap, failed / skipped runs
    devices/<target>.html one per device: specs, ranking table, FPS / latency breakdown / RAM charts
    models/<model>.html   one per model: model info, per-device table, FPS / RAM / detections charts
    method.html           how the simulation works and what it does not model
    print.html            all pages above in one document for the browser's Print -> Save as PDF
Charts and base CSS come from html_report.py.

Sweep suites: every pair shows its BEST configuration (source resolution, model input size, confidence) with the reason it was
picked (manual rating or the automatic real-time + stable rule); the device pages link to the per-run report that lists every
configuration, and the model pages hold a per-configuration table with the estimated FPS on every device.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.benchmark import sweep as sweep_mod
from src.benchmark import sweep_html
from src.benchmark.html_report import CSS, _chart_card, _e, _f, _kpi, _legend, hbars, stacked_bars
from src.benchmark.report_builder import ACCURACY_DISCLAIMER, METHOD_NOTE
from src.benchmark.suite import (AMBER_FPS, RAM_FAIL_PCT, RAM_WARN_PCT, _atomic_write, cell_note, heat, iter_cells,
                                 load_run_report, required_fps_of, summarize)

SITE_CSS = """
:root { --na-bg:#e2e8f0; --link:#1d4ed8; }
@media (prefers-color-scheme: dark) { :root { --na-bg:#1e293b; --link:#93c5fd; } }
@media print { :root { --na-bg:#e2e8f0; --link:#1d4ed8; } }
a { color:var(--link); }
.wrap { max-width:1240px; }
.topnav { display:flex; flex-wrap:wrap; gap:4px 14px; align-items:center; padding:0 0 12px; margin-bottom:12px;
  border-bottom:1px solid var(--border); font-size:.9rem; }
.topnav a, .topnav summary { color:var(--text); text-decoration:none; cursor:pointer; padding:4px 10px; border-radius:8px;
  list-style:none; display:inline-block; }
.topnav summary::-webkit-details-marker { display:none; }
.topnav a:hover, .topnav summary:hover { background:var(--card); }
.topnav .active { background:var(--card); font-weight:700; box-shadow:inset 0 -2px 0 var(--host); }
.topnav .brand { font-weight:800; margin-right:8px; }
.topnav .sp { flex:1; }
.dd { position:relative; }
.dd .menu { position:absolute; z-index:20; top:100%; left:0; min-width:250px; max-height:70vh; overflow:auto; background:var(--card);
  border:1px solid var(--border); border-radius:10px; padding:6px; box-shadow:0 8px 24px rgba(0,0,0,.18); }
.dd .menu a { display:block; white-space:nowrap; }
.crumbs { font-size:.8rem; color:var(--muted); margin-bottom:8px; }
.crumbs a { text-decoration:none; }
.pager { display:flex; justify-content:space-between; gap:12px; margin:22px 0 6px; font-size:.9rem; }
.pager a { text-decoration:none; padding:6px 12px; border:1px solid var(--border); border-radius:8px; background:var(--card); }
.foot { margin-top:26px; font-size:.75rem; color:var(--muted); border-top:1px solid var(--border); padding-top:10px; }
.banner { border-radius:10px; padding:10px 14px; margin-bottom:14px; font-size:.9rem; border:1px solid var(--border); }
.banner.warn { background:var(--warn-bg); color:var(--warn); } .banner.fail { background:var(--fail-bg); color:var(--fail); }
.scroll { overflow-x:auto; }
table.matrix { width:100%; border-collapse:separate; border-spacing:3px; }
table.matrix th, table.matrix td { border-bottom:none; }
table.matrix th { text-align:center; vertical-align:bottom; text-transform:none; letter-spacing:0; font-size:.82rem; color:var(--text); }
table.matrix th a { text-decoration:none; } table.matrix th a:hover { text-decoration:underline; }
table.matrix th.m { text-align:left; white-space:nowrap; font-weight:600; }
table.matrix td.hm { text-align:center; border-radius:6px; padding:8px 8px; white-space:nowrap; font-variant-numeric:tabular-nums;
  font-weight:700; }
td.hm small { display:block; font-weight:500; font-size:.68rem; opacity:.85; }
td.hm a { color:inherit; text-decoration:none; } td.hm a:hover { text-decoration:underline; }
.h-green { background:var(--ok-bg); color:var(--ok); } .h-amber { background:var(--warn-bg); color:var(--warn); }
.h-red { background:var(--fail-bg); color:var(--fail); }
.h-failed { background:var(--na-bg); color:var(--fail); } .h-skipped, .h-cancelled, .h-pending, .h-running, .h-na
  { background:var(--na-bg); color:var(--muted); font-weight:500 !important; }
.sw { display:inline-block; width:14px; height:14px; border-radius:4px; margin-right:5px; vertical-align:-2px; }
svg .fill-ok { fill:var(--ok); } svg .fill-warn { fill:var(--warn); } svg .fill-fail { fill:var(--fail); }
.charts .card.wide { grid-column:1 / -1; }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; margin-bottom:16px; }
.two .card { margin-bottom:0; }
.s { color:var(--muted); font-size:.8rem; }
.nm { font-size:.72rem; font-weight:400; margin-left:3px; opacity:.85; }
ul.tight { margin:4px 0 0; padding-left:18px; } ul.tight li { margin:2px 0; }
td.msg { max-width:420px; font-size:.78rem; white-space:normal; }
.printpage { break-before:page; }
td.hm small.cfg { font-weight:600; opacity:1; }
table.cfgtable td.hm { padding:4px 8px; font-size:.8rem; }
td.star { font-weight:700; }
.toc a { display:block; }
@media print {
  .topnav, .pager, .noprint { display:none !important; }
  .wrap { max-width:none; }
  table.matrix td.hm a { text-decoration:none; }
  h1, h2 { break-after:avoid; }
  .printpage:first-of-type { break-before:auto; }
}
"""

STATUS_LABEL = {"failed": "failed", "skipped": "skipped", "cancelled": "cancelled", "pending": "pending", "running": "running"}
SEV_CSS = {"green": "ok", "amber": "warn", "red": "fail"}
DECODED_EXCLUDED = ("unsupported", "unknown")


class Raw(str):
    """HTML that must not be escaped again."""


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def short_model(name: str) -> str:
    return Path(name).stem


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h {s % 3600 // 60:02d}m"
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _kv(title: str, rows: List[Tuple[str, Any]]) -> str:
    """Key/value card; Raw values are inserted as HTML, everything else is escaped."""
    body = "".join(f"<tr><td>{_e(k)}</td><td>{v if isinstance(v, Raw) else _e(v if isinstance(v, str) else _f(v))}</td></tr>"
                   for k, v in rows)
    return f'<div class="card"><h2>{_e(title)}</h2><table class="kv">{body}</table></div>'


class Ctx:
    """Read-only view of a manifest with the lookups the pages share."""

    def __init__(self, manifest: Dict[str, Any], folder: Optional[Path] = None):
        self.m = manifest
        self.folder = folder
        self._reports: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        cfg = manifest["config"]
        self.cfg = cfg
        self.models: List[str] = cfg["models"]
        self.targets: List[str] = cfg["targets"]
        self.required = required_fps_of(manifest)
        self.ti: Dict[str, Any] = manifest["targets_info"]
        self.mi: Dict[str, Any] = manifest["models_info"]
        stems = [Path(x).stem for x in self.models]
        self.model_slug = {m: slug(Path(m).stem if stems.count(Path(m).stem) == 1 else m) for m in self.models}
        self.dev_slug = {t: slug(t) for t in self.targets}
        self.summary = summarize(manifest)

    def cell(self, model: str, target: str) -> Dict[str, Any]:
        return self.m["cells"].get(model, {}).get(target) or {"status": "pending"}

    def best(self, model: str, target: str) -> Optional[Dict[str, Any]]:
        """Best-configuration summary of a finished pair (None for suites without a sweep)."""
        return self.cell(model, target).get("sweep") or None

    def report(self, model: str, target: str) -> Optional[Dict[str, Any]]:
        """report.json of a pair's run when it has sweep data (all configurations)."""
        key = (model, target)
        if key not in self._reports:
            c = self.cell(model, target)
            r = load_run_report(self.folder, c) if self.folder is not None and c.get("sweep") else None
            self._reports[key] = r if r and r.get("sweep") else None
        return self._reports[key]

    @property
    def has_sweep(self) -> bool:
        return bool(self.cfg.get("sweep")) or any(c.get("sweep") for _, _, c in iter_cells(self.m))

    def note(self, model: str, target: str) -> str:
        """User note of the model/device pair ("" when none; suites without notes have none)."""
        return cell_note(self.m, model, target)

    def device(self, target: str) -> str:
        return (self.ti.get(target) or {}).get("device") or target

    def simulated(self, target: str) -> bool:
        return bool((self.ti.get(target) or {}).get("simulated"))

    def ok_cells(self, target: Optional[str] = None, model: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        return [(m, t, c) for m, t, c in iter_cells(self.m)
                if c.get("status") == "ok" and (target is None or t == target) and (model is None or m == model)]

    def heat(self, cell: Dict[str, Any]) -> Optional[str]:
        return heat(cell, self.required)


class Links:
    """Link targets differ between the multipage site (files) and print.html (anchors in one document)."""

    def __init__(self, ctx: Ctx, print_mode: bool = False, prefix: str = ""):
        self.ctx, self.print_mode, self.prefix = ctx, print_mode, prefix

    def home(self) -> str:
        return "#overview" if self.print_mode else f"{self.prefix}index.html"

    def method(self) -> str:
        return "#method" if self.print_mode else f"{self.prefix}method.html"

    def device(self, target: str) -> str:
        s = self.ctx.dev_slug[target]
        return f"#device-{s}" if self.print_mode else f"{self.prefix}devices/{s}.html"

    def model(self, model: str) -> str:
        s = self.ctx.model_slug[model]
        return f"#model-{s}" if self.print_mode else f"{self.prefix}models/{s}.html"

    def run(self, report_id: str) -> str:
        return f"{self.prefix}runs/{report_id}/report.html"


# ---------------------------------------------------------------- shared building blocks

def _note_td(note: str) -> str:
    """Notes cell of a table row: always present, empty when the pair has no note."""
    return f'<td class="notes" dir="auto">{_e(note)}</td>'


def _mark(note: str) -> str:
    """Small marker in a heatmap cell that has a note (the text itself is in the tooltip)."""
    return '<span class="nm" title="has a note">&#9998;</span>' if note else ""


def _with_note(tip: str, note: str) -> str:
    return f"{tip}\nNote: {note}" if note else tip


def _status_cell(c: Dict[str, Any], note: str = "") -> str:
    st = c.get("status", "pending")
    return (f'<td class="hm h-{_e(st)}" title="{_e(_with_note(str(c.get("error") or st), note))}">'
            f'{_e(STATUS_LABEL.get(st, st))}{_mark(note)}</td>')


def _cfg_small(c: Dict[str, Any]) -> str:
    """Best configuration under a heatmap value (a star when it was picked by a manual rating)."""
    b = c.get("sweep")
    if not b:
        return ""
    return f'<small class="cfg" title="best configuration">{_e(b["short"])}{" &#9733;" if b.get("rated") else ""}</small>'


def _cfg_tip(c: Dict[str, Any]) -> str:
    b = c.get("sweep")
    if not b:
        return ""
    lines = [f"Best configuration: {b['label'] if b.get('label') else b['short']}", f"Why: {b['reason']}"]
    if b.get("stability") is not None:
        lines.append(f"Stability {_f(b['stability'], 2)} (agreement F1 {_f(b.get('agreement_f1'), 2)}, temporal {_f(b.get('temporal'), 2)})")
    if b.get("coverage") is not None or b.get("duplicates") is not None:
        lines.append(f"Ratings: coverage {_f(b.get('coverage'))}/5, duplicates {_f(b.get('duplicates'))}")
    if b.get("auto_pick"):
        lines.append(f"Automatic pick: {b['auto_pick']}")
    return "\n".join(lines)


def _tip(c: Dict[str, Any], note: str = "") -> str:
    tip = f"P95 {_f(c.get('latency_p95_ms'), 1)} ms | RAM {_f(c.get('peak_ram_mb'), 0)} MB"
    if c.get("ram_pct") is not None:
        tip += f" ({_f(c['ram_pct'], 0)}% of budget)"
    tip += f" | {str(c.get('verdict')).upper()}"
    if c.get("sweep"):
        tip += "\n" + _cfg_tip(c)
    return _with_note(tip, note)


def _fps_cell(ctx: Ctx, L: Links, model: str, target: str) -> str:
    c = ctx.cell(model, target)
    note = ctx.note(model, target)
    if c.get("status") != "ok":
        return _status_cell(c, note)
    host = "" if c.get("simulated") else "<small>host-measured</small>"
    return (f'<td class="hm h-{ctx.heat(c)}" title="{_e(_tip(c, note))}"><a href="{_e(L.run(c["report_id"]))}">'
            f'{_e(_f(c["fps"], 1))}</a>{_mark(note)}{_cfg_small(c)}{host}</td>')


def _rt_cell(ctx: Ctx, L: Links, model: str, target: str) -> str:
    c = ctx.cell(model, target)
    note = ctx.note(model, target)
    if c.get("status") != "ok":
        return _status_cell(c, note)
    cap = c.get("realtime_capable")
    return (f'<td class="hm h-{"green" if cap else "red"}" title="{_e(_tip(c, note))}"><a href="{_e(L.run(c["report_id"]))}">'
            f'{"&#10003;" if cap else "&#10007;"}</a>{_mark(note)}<small>{_e(_f(c.get("realtime_factor"), 2))}x</small>{_cfg_small(c)}</td>')


def _ram_class(pct: Optional[float]) -> Optional[str]:
    if pct is None:
        return None
    return "green" if pct < RAM_WARN_PCT else ("amber" if pct < RAM_FAIL_PCT else "red")


def _ram_cell(ctx: Ctx, L: Links, model: str, target: str) -> str:
    c = ctx.cell(model, target)
    note = ctx.note(model, target)
    if c.get("status") != "ok":
        return _status_cell(c, note)
    cls = _ram_class(c.get("ram_pct"))
    tip = _with_note(f"peak {_f(c.get('peak_ram_mb'), 0)} MB of {_f(c.get('ram_limit_mb'), 0)} MB budget", note)
    if cls is None:
        return (f'<td class="hm h-na" title="{_e(tip)}">{_e(_f(c.get("peak_ram_mb"), 0))} MB{_mark(note)}'
                f'<small>no budget</small></td>')
    return (f'<td class="hm h-{cls}" title="{_e(tip)}"><a href="{_e(L.run(c["report_id"]))}">{_e(_f(c["ram_pct"], 0))}%</a>{_mark(note)}'
            f'<small>{_e(_f(c.get("peak_ram_mb"), 0))} MB</small>{_cfg_small(c)}</td>')


def _matrix(ctx: Ctx, L: Links, cell_fn) -> str:
    head = "".join(
        f'<th><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a><br><span class="s">{_e(t)}'
        f'{"" if ctx.simulated(t) else " &middot; host-measured"}</span></th>' for t in ctx.targets)
    rows = "".join(
        f'<tr><th class="m"><a href="{_e(L.model(m))}">{_e(short_model(m))}</a></th>'
        f'{"".join(cell_fn(ctx, L, m, t) for t in ctx.targets)}</tr>' for m in ctx.models)
    return f'<div class="scroll"><table class="matrix"><tr><th></th>{head}</tr>{rows}</table></div>'


def _sw(cls: str, text: str) -> str:
    return f'<span><i class="sw h-{cls}"></i>{_e(text)}</span>'


def _fps_legend(ctx: Ctx) -> str:
    return ('<div class="legend">' + _sw("green", f"at or above the required {ctx.required:g} FPS (real-time)")
            + _sw("amber", f"at least {AMBER_FPS:g} FPS") + _sw("red", f"below {AMBER_FPS:g} FPS")
            + _sw("failed", "failed / skipped / not run") + "</div>")


def _kind_note(ctx: Ctx) -> str:
    host_only = [t for t in ctx.targets if not ctx.simulated(t)]
    txt = ("FPS is the estimated on-device FPS (host-measured processing time scaled by the device's calibration anchor). "
           "Hover a cell for P95 latency, RAM and verdict; click it for the full run report.")
    if host_only:
        txt += f" {', '.join(host_only)} is not simulated: its numbers are measured on this host and are not comparable to the estimates."
    if ctx.has_sweep:
        txt += (" Every cell shows its best configuration (source resolution · model input size · confidence) under the value; a star means it was "
                "picked by your manual rating, otherwise by the automatic real-time + stable rule (hover for the reason).")
    return txt


def _status_badge(status: str) -> str:
    cls = {"completed": "ok", "running": "warn", "stopped": "warn", "failed": "fail"}.get(status, "warn")
    return f'<span class="badge {cls}">{_e(status.upper())}</span>'


def _verdict_badge(v: Optional[str]) -> str:
    return f'<span class="pill {_e(v or "warn")}">{_e(str(v or "n/a").upper())}</span>'


def _cfg_inline(c: Dict[str, Any]) -> str:
    b = c.get("sweep")
    return f' &middot; {_e(b["short"])}{" &#9733;" if b.get("rated") else ""}' if b else ""


def _problem_rows(ctx: Ctx, target: Optional[str] = None, model: Optional[str] = None) -> str:
    rows = []
    for m, t, c in iter_cells(ctx.m):
        if c.get("status") == "ok" or (target and t != target) or (model and m != model):
            continue
        rows.append(f'<tr><td>{_e(short_model(m))}</td><td>{_e(t)}</td><td>{_e(c.get("status", "pending"))}</td>'
                    f'<td class="msg">{_e(str(c.get("error") or "")[:400])}</td>{_note_td(ctx.note(m, t))}</tr>')
    if not rows:
        return ""
    return ('<div class="card"><h2>Failed, skipped or not run</h2><div class="scroll"><table><tr><th>Model</th><th>Device</th>'
            f'<th>Status</th><th>Reason</th><th>Notes</th></tr>{"".join(rows)}</table></div></div>')


def _pct(c: Dict[str, Any]) -> str:
    return "n/a" if c.get("ram_pct") is None else f"{_f(c['ram_pct'], 0)}%"


def _ram_txt(c: Dict[str, Any]) -> str:
    lim = f" / {_f(c['ram_limit_mb'], 0)}" if c.get("ram_limit_mb") else ""
    return f"{_f(c.get('peak_ram_mb'), 0)}{lim} MB"


# ---------------------------------------------------------------- page bodies

def overview_body(ctx: Ctx, L: Links) -> str:  # noqa: C901
    m, cfg, s = ctx.m, ctx.cfg, ctx.summary
    counts = s["counts"]
    ok_cells = ctx.ok_cells()
    rt_ok = sum(1 for _, _, c in ok_cells if c.get("realtime_capable"))
    best = s["best"]
    video = Path(cfg["video"]).name if cfg.get("video") else "synthetic frames"
    banner = ""
    if m["status"] != "completed":
        missing = counts["pending"] + counts["running"] + counts["cancelled"]
        why = {"running": "This suite is still running: cells fill in as they finish.",
               "stopped": f"This suite was stopped before it finished: {missing} run(s) were cancelled. Resume it from the web app or with "
                          f"<code>--resume {_e(m['suite_id'])}</code>.",
               "failed": "No run of this suite produced a result."}.get(m["status"], "")
        banner = f'<div class="banner {"fail" if m["status"] == "failed" else "warn"}">{why}</div>'

    kpis = "".join([
        _kpi("Runs succeeded", f"{counts['ok']}/{s['total'] - counts['skipped']}", "",
             f"{counts['failed']} failed, {counts['skipped']} skipped, {counts['cancelled'] + counts['pending'] + counts['running']} not run",
             "ok" if counts["failed"] == 0 and not s["incomplete"] else ("fail" if counts["ok"] == 0 else "warn")),
        _kpi("Real-time capable", f"{rt_ok}/{len(ok_cells)}", "pairs", f"reach the required {ctx.required:g} FPS",
             "ok" if ok_cells and rt_ok == len(ok_cells) else ("warn" if rt_ok else "fail")),
        _kpi("Fastest pair", _f(best["fps"], 1) if best else "n/a", "FPS",
             f"{short_model(best['model'])} on {best['target']}" if best else "no successful run"),
        _kpi("Suite time", fmt_duration(m["timings"].get("elapsed_s")), "", f"{cfg['frames']} frames per run, warmup {cfg['warmup']}"),
    ])

    head = (f'<div class="head"><div><h1>Benchmark suite: {len(ctx.models)} models x {len(ctx.targets)} devices</h1>'
            f'<div class="sub">{_e(m["created"].replace("T", " "))} &middot; <span class="mono">{_e(m["suite_id"])}</span> &middot; '
            f'{cfg["frames"]} frames per run &middot; source: {_e(video)} &middot; required {ctx.required:g} FPS &middot; '
            f'executed {"in Docker (CPU/RAM-limited containers)" if cfg.get("execution") == "docker" else "on the host"}</div></div>'
            f'{_status_badge(m["status"])}</div>')

    sw_cfg = cfg.get("sweep")
    sweep_note = ""
    if sw_cfg:
        sweep_note = (f'<div class="card"><h2>Sweep</h2><p>Every model/device pair was tested over source resolutions '
                      f'{_e(", ".join("native" if h == 0 else str(h) + "p" for h in sw_cfg["source_heights"]))}, model input sizes '
                      f'{_e(", ".join(str(x) for x in sw_cfg["input_sizes"]) or "model default")} (only for models with a dynamic input; others are locked to their own size) '
                      f'and confidence thresholds {_e(", ".join(sweep_mod.conf_text(x) for x in sw_cfg["conf_thresholds"]))}. The overview shows the best '
                      f'configuration of each pair and why it was picked; the run report of a pair lists all configurations, the model pages compare them across devices. '
                      f'Rate the detection quality in the web app (suite card, &#11088; Rate detection quality) to make your judgement decide; see '
                      f'<a href="{_e(L.method())}">Methodology</a>.</p></div>')
    fps = (f'<div class="card" id="fps-heatmap"><h2>Estimated FPS: model x device</h2>{_fps_legend(ctx)}'
           f'{_matrix(ctx, L, _fps_cell)}<p class="note">{_e(_kind_note(ctx))} A &#9998; marks a model/device pair with a note: '
           f'hover the cell to read it (all notes are in the Notes column of the device and model pages).</p></div>')

    best_rows = []
    for t in ctx.targets:
        oks = ctx.ok_cells(target=t)
        if not oks:
            best_rows.append(f'<tr><td><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a></td>'
                             f'<td colspan="3" class="s">no successful run</td>{_note_td("")}</tr>')
            continue
        fastest = max(oks, key=lambda x: x[2]["fps"])
        decoded = [x for x in oks if x[2].get("output_format") not in DECODED_EXCLUDED]
        det = max(decoded, key=lambda x: x[2]["fps"]) if decoded else None
        rt = sum(1 for x in oks if x[2].get("realtime_capable"))
        det_html = "n/a"
        if det:
            det_html = (f'<a href="{_e(L.model(det[0]))}">{_e(short_model(det[0]))}</a> '
                        f'<span class="s">{_e(_f(det[2]["fps"], 1))} FPS{_cfg_inline(det[2])}</span>')
        picked = [fastest[0]] + ([det[0]] if det and det[0] != fastest[0] else [])
        notes_html = "".join(f'<div><span class="s">{_e(short_model(pm))}:</span> {_e(ctx.note(pm, t))}</div>'
                             for pm in picked if ctx.note(pm, t))
        best_rows.append(
            f'<tr><td><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a><br><span class="s">{_e(t)}</span></td>'
            f'<td><a href="{_e(L.model(fastest[0]))}">{_e(short_model(fastest[0]))}</a> '
            f'<span class="s">{_e(_f(fastest[2]["fps"], 1))} FPS{_cfg_inline(fastest[2])}</span></td>'
            f'<td>{det_html}</td><td class="num">{rt}/{len(oks)}</td><td class="notes" dir="auto">{notes_html}</td></tr>')
    best_tbl = ('<div class="card"><h2>Best model per device</h2><div class="scroll"><table><tr><th>Device</th><th>Fastest model</th>'
                '<th>Fastest with decoded detections</th><th class="num">Real-time models</th><th>Notes</th></tr>'
                f'{"".join(best_rows)}</table></div><p class="note">Some models (face, pose, embedding) have an output format this harness does '
                'not decode: they are timed correctly but report 0 detections, so the second column ignores them.</p></div>')

    rt_card = (f'<div class="card"><h2>Real-time capable ({ctx.required:g} FPS required)</h2>'
               f'<div class="legend">{_sw("green", "capable")}{_sw("red", "not real-time")}{_sw("failed", "no result")}</div>'
               f'{_matrix(ctx, L, _rt_cell)}<p class="note">The small number is the real-time factor: estimated FPS divided by the required FPS.</p></div>')

    ram_card = (f'<div class="card"><h2>Peak RAM as % of the device budget</h2>'
                f'<div class="legend">{_sw("green", f"below {RAM_WARN_PCT:g}%")}{_sw("amber", f"{RAM_WARN_PCT:g} to {RAM_FAIL_PCT:g}%")}'
                f'{_sw("red", f"{RAM_FAIL_PCT:g}% or more (thin headroom)")}{_sw("failed", "no result")}</div>'
                f'{_matrix(ctx, L, _ram_cell)}<p class="note">Measured peak resident memory of the benchmark process inside the container '
                'divided by the RAM budget of that device in targets.yaml.</p></div>')

    note = (f'<div class="card"><h2>About these numbers</h2><p>{_e(METHOD_NOTE)}</p><p class="note">{_e(ACCURACY_DISCLAIMER)} '
            f'See <a href="{_e(L.method())}">Methodology</a>.</p></div>')
    return f'{head}{banner}<div class="kpis">{kpis}</div>{sweep_note}{fps}{best_tbl}{rt_card}{ram_card}{_problem_rows(ctx)}{note}'


def _device_specs(ctx: Ctx, t: str) -> str:
    ti = ctx.ti.get(t) or {}
    cal = ti.get("calibration")
    cont = ti.get("container") or {}
    rows: List[Tuple[str, Any]] = [
        ("Key", t), ("Device", ti.get("device")), ("Description", ti.get("description")),
        ("Compute unit", ti.get("compute_unit")), ("Runtime modelled", ti.get("runtime")),
        ("CPU cores", ti.get("cpu_cores") or "all"),
        ("CPU single-core score", ti.get("cpu_single_core_score")),
        ("CPU scale (host to device)", ti.get("cpu_scale")),
        ("RAM / VRAM budget", f"{_f(ti.get('ram_limit_mb'), 0)} / {_f(ti.get('vram_limit_mb'), 0)} MB"),
        ("Timing basis", "estimated from a calibration anchor" if ti.get("simulated") else "measured on the host (not simulated)"),
    ]
    if cal:
        rows += [("Calibration anchor", f"{cal['model']} = {cal['latency_ms']} ms ({cal['gflops']} GFLOPs)"),
                 ("Anchor source", Raw(_link_text(cal.get("source"))))]
    if cont:
        rows.append(("Container limits (measured)", f"{_f(cont.get('cpu_limit_cores'))} cores / {_f(cont.get('memory_limit_mb'), 0)} MB "
                                                     f"(host shows {cont.get('logical_cpus')} logical CPUs)"))
    return _kv("Device", rows)


def _link_text(url: Optional[str]) -> str:
    if url and url.startswith("http"):
        return f'<a href="{_e(url)}" rel="noopener">{_e(url)}</a>'
    return _e(url or "n/a")


def device_body(ctx: Ctx, L: Links, t: str) -> str:
    ti = ctx.ti.get(t) or {}
    sim = ctx.simulated(t)
    oks = sorted(ctx.ok_cells(target=t), key=lambda x: -x[2]["fps"])
    head = (f'<div class="head"><div><h1>{_e(ctx.device(t))}</h1><div class="sub">{_e(t)} &middot; {_e(ti.get("runtime"))} &middot; '
            f'{"estimated timing" if sim else "host-measured timing (not simulated)"}</div></div></div>')

    rows = []
    for rank, (m, _, c) in enumerate(oks, 1):
        b = c.get("sweep")
        rows.append(
            f'<tr><td class="num">{rank}</td><td><a href="{_e(L.model(m))}">{_e(short_model(m))}</a></td>'
            + (f'<td>{_e(b["short"])}{" &#9733;" if b.get("rated") else ""}<br><span class="s">stability {_e(_f(b.get("stability"), 2))} &middot; '
               f'{_e(b["n_configs"])} configs</span></td>' if ctx.has_sweep and b else ('<td class="s">n/a</td>' if ctx.has_sweep else ""))
            + f'<td class="num h-{ctx.heat(c)}"><b>{_e(_f(c["fps"], 1))}</b></td>'
            f'<td class="num">{_e(_f(c.get("est_latency_p50_ms") if sim else c.get("host_latency_p50_ms"), 1))}</td>'
            f'<td class="num">{_e(_f(c.get("est_latency_p95_ms") if sim else c.get("host_latency_p95_ms"), 1))}</td>'
            f'<td class="num">{_e(_f(c.get("cpu_side_ms"), 1))}</td><td class="num">{_e(_f(c.get("inference_ms"), 1))}</td>'
            f'<td class="num">{_e(_f(c.get("host_fps"), 1))}</td><td class="num">{_e(_ram_txt(c))} ({_e(_pct(c))})</td>'
            f'<td class="num">{_e(_f(c.get("dets_per_frame"), 2))}</td><td>{_e(c.get("output_format"))}</td>'
            f'<td>{_verdict_badge(c.get("verdict"))}</td>{_note_td(ctx.note(m, t))}'
            + (f'<td class="msg">{_e(b["reason"])}</td>' if ctx.has_sweep and b else ('<td></td>' if ctx.has_sweep else ""))
            + f'<td><a href="{_e(L.run(c["report_id"]))}">{"all configurations" if b else "report"}</a></td></tr>')
    basis = "Est. FPS" if sim else "Host FPS"
    cfg_th = '<th>Best configuration</th>' if ctx.has_sweep else ""
    why_th = '<th>Why this configuration</th>' if ctx.has_sweep else ""
    table = ('<div class="card"><h2>Model ranking</h2><div class="scroll"><table><tr><th class="num">#</th><th>Model</th>'
             f'{cfg_th}<th class="num">{basis}</th><th class="num">P50 ms</th><th class="num">P95 ms</th><th class="num">CPU-side ms</th>'
             '<th class="num">Inference ms</th><th class="num">Host FPS</th><th class="num">Peak RAM / budget</th>'
             f'<th class="num">Dets/frame</th><th>Output</th><th>Verdict</th><th>Notes</th>{why_th}<th></th></tr>'
             f'{"".join(rows) or "<tr><td colspan=17 class=s>No successful run on this device.</td></tr>"}</table></div>'
             '<p class="note">CPU-side = pre + post-processing on the device CPU; Inference = accelerator (or CPU) time for the network. '
             'Latencies are per frame' + (' and estimated.' if sim else ' and measured on the host.')
             + (' Every row is the best configuration of that model on this device (source resolution · model input size · confidence; a star = picked by '
                'your manual rating); the linked run report lists all configurations.' if ctx.has_sweep else '') + '</p></div>')

    charts = ""
    if oks:
        fps_svg = hbars([(short_model(m), c["fps"], SEV_CSS[ctx.heat(c) or "red"]) for m, _, c in oks],
                        "Estimated FPS" if sim else "Host FPS (measured)", "FPS by model", limit=len(oks),
                        label_chars=28, label_w=200, marker=ctx.required, marker_label=f"required {ctx.required:g} FPS")
        stack_rows = [(short_model(m), [("CPU-side (pre+post)", c.get("cpu_side_ms") or 0.0, "pre"),
                                        ("Inference", c.get("inference_ms") or 0.0, "dev" if sim else "host")])
                      for m, _, c in sorted(oks, key=lambda x: -((x[2].get("cpu_side_ms") or 0) + (x[2].get("inference_ms") or 0)))]
        stack_svg = stacked_bars(stack_rows, "Latency breakdown per model", label_w=200, label_chars=28)
        budget = ti.get("ram_limit_mb")
        ram_svg = hbars([(short_model(m), c.get("peak_ram_mb") or 0.0, SEV_CSS.get(_ram_class(c.get("ram_pct")) or "", "host"))
                         for m, _, c in sorted(oks, key=lambda x: -(x[2].get("peak_ram_mb") or 0))],
                        "Peak RAM (MB)", "Peak RAM by model", limit=len(oks), label_chars=28, label_w=200,
                        marker=budget, marker_label=f"budget {_f(budget, 0)} MB" if budget else "")
        charts = (
            '<div class="charts">'
            + _chart_card("Throughput by model", _legend([("meets required FPS", "ok"), (f"at least {AMBER_FPS:g} FPS", "warn"), ("slower", "fail")]),
                          fps_svg, "Dashed line: required FPS. Longer is better.")
            + _chart_card("Peak RAM by model", _legend([(f"below {RAM_WARN_PCT:g}% of budget", "ok"), ("thin headroom", "warn"), ("very thin", "fail")]),
                          ram_svg, "Dashed line: the RAM budget of this device." if budget else "This device has no RAM budget.")
            + '<div class="card wide"><h2>Latency breakdown per frame</h2>'
            + _legend([("CPU-side (pre + post-processing)", "pre"), ("Inference (device est.)" if sim else "Inference (host)", "dev" if sim else "host")])
            + stack_svg + '<p class="note">When the CPU-side part dominates, a faster accelerator would not help: the CPU, not the network, is the bottleneck.</p></div>'
            + '</div>')
    return f'{head}<div class="tables" style="margin-bottom:16px">{_device_specs(ctx, t)}</div>{table}{charts}{_problem_rows(ctx, target=t)}'


def model_body(ctx: Ctx, L: Links, model: str) -> str:
    mi = ctx.mi.get(model) or {}
    oks = ctx.ok_cells(model=model)
    fmt_name = next((c.get("output_format") for _, _, c in oks), mi.get("output_format") or "n/a")
    classes = mi.get("class_names") or []
    class_txt = ", ".join(classes[:24]) + (f" ... (+{len(classes) - 24} more)" if len(classes) > 24 else "") if classes else "not stored in the model"
    info = _kv("Model", [
        ("File", model), ("Format", mi.get("format")), ("Size", f"{_f(mi.get('size_mb'))} MB"),
        ("Compute", f"{_f(mi.get('gflops'), 3)} GFLOPs, {_f(mi.get('params_m'), 3)} M params" if mi.get("gflops") is not None else "n/a"),
        ("Input size (h x w)", " x ".join(str(x) for x in mi["input_size"]) if mi.get("input_size") else "n/a"),
        ("Output format", fmt_name), ("Classes", class_txt), ("SHA-256 (first 16)", mi.get("sha256") or "n/a"),
    ])
    head = (f'<div class="head"><div><h1>{_e(short_model(model))}</h1><div class="sub">{_e(model)} &middot; '
            f'{len(oks)} of {len(ctx.targets)} devices benchmarked</div></div></div>')
    warn = ""
    if fmt_name in DECODED_EXCLUDED:
        warn = ('<div class="banner warn">The output format of this model is not decoded by the harness: latency and RAM are real, '
                'but 0 detections says nothing about what the model finds.</div>')
    if not oks:
        warn += '<div class="banner fail">This model produced no result on any device in this suite; see the reasons below.</div>'

    rows = []
    for t in ctx.targets:
        c = ctx.cell(model, t)
        if c.get("status") != "ok":
            rows.append(f'<tr><td><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a></td><td colspan="{12 if ctx.has_sweep else 10}" class="s">'
                        f'{_e(c.get("status", "pending"))}: {_e(str(c.get("error") or "")[:200])}</td>{_note_td(ctx.note(model, t))}</tr>')
            continue
        sim = c.get("simulated")
        issues = [i for i in c.get("verdict_items", []) if i["level"] != "ok"]
        msgs = ('<ul class="tight">' + "".join(f'<li>{_e(i["message"])}</li>' for i in issues) + "</ul>") if issues else '<span class="s">all checks passed</span>'
        b = c.get("sweep")
        rows.append(
            f'<tr><td><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a><br><span class="s">{_e(t)}{"" if sim else " &middot; host-measured"}</span></td>'
            + (f'<td>{_e(b["short"])}{" &#9733;" if b.get("rated") else ""}</td>' if ctx.has_sweep and b else ('<td class="s">n/a</td>' if ctx.has_sweep else ""))
            + f'<td class="num h-{ctx.heat(c)}"><b>{_e(_f(c["fps"], 1))}</b></td>'
            f'<td class="num">{_e(_f(c.get("latency_p50_ms"), 1))}</td><td class="num">{_e(_f(c.get("latency_p95_ms"), 1))}</td>'
            f'<td class="num">{_e(_f(c.get("cpu_side_ms"), 1))} / {_e(_f(c.get("inference_ms"), 1))}</td>'
            f'<td class="num">{_e(_f(c.get("host_fps"), 1))}</td><td class="num">{_e(_ram_txt(c))} ({_e(_pct(c))})</td>'
            f'<td class="num">{_e(_f(c.get("dets_per_frame"), 2))}<br><span class="s">{_e(_f(c.get("frames_with_detections_pct"), 0))}% of frames</span></td>'
            f'<td class="num">{_e(_f(c.get("cold_start_ms"), 0))} / {_e(_f(c.get("first_inference_ms"), 0))}</td>'
            f'<td>{_verdict_badge(c.get("verdict"))}</td><td class="msg">{msgs} <a href="{_e(L.run(c["report_id"]))}">{"all configurations" if b else "run report"}</a></td>'
            + (f'<td class="msg">{_e(b["reason"])}</td>' if ctx.has_sweep and b else ('<td></td>' if ctx.has_sweep else ""))
            + f'{_note_td(ctx.note(model, t))}</tr>')
    table = ('<div class="card"><h2>Results per device (best configuration)</h2><div class="scroll"><table><tr><th>Device</th>'
             + ('<th>Best configuration</th>' if ctx.has_sweep else '') + '<th class="num">FPS</th>'
             '<th class="num">P50 ms</th><th class="num">P95 ms</th><th class="num">CPU-side / inference ms</th><th class="num">Host FPS</th>'
             '<th class="num">Peak RAM / budget</th><th class="num">Dets/frame</th><th class="num">Cold start ms (load / 1st inference)</th>'
             f'<th>Verdict</th><th>Checks</th>{"<th>Why this configuration</th>" if ctx.has_sweep else ""}<th>Notes</th></tr>{"".join(rows)}</table></div>'
             '<p class="note">FPS, P50 and P95 are estimated on-device figures for simulated devices and measured host figures for host-measured ones.</p></div>')

    charts = ""
    if oks:
        fps_svg = hbars([(t, c["fps"], SEV_CSS[ctx.heat(c) or "red"]) for _, t, c in sorted(oks, key=lambda x: -x[2]["fps"])],
                        "FPS (estimated; host-measured for x86-cpu)", "FPS by device", limit=len(oks), label_chars=20, label_w=130,
                        marker=ctx.required, marker_label=f"required {ctx.required:g} FPS")
        ram_svg = hbars([(t, c.get("peak_ram_mb") or 0.0, SEV_CSS.get(_ram_class(c.get("ram_pct")) or "", "host")) for _, t, c in oks],
                        "Peak RAM (MB)", "Peak RAM by device", limit=len(oks), label_chars=20, label_w=130)
        det_svg = hbars([(t, float(c.get("dets_per_frame") or 0.0), "host") for _, t, c in oks],
                        "Detections per frame", "Detections by device", limit=len(oks), label_chars=20, label_w=130)
        charts = ('<div class="charts">'
                  + _chart_card("Throughput by device", _legend([("meets required FPS", "ok"), (f"at least {AMBER_FPS:g} FPS", "warn"), ("slower", "fail")]),
                                fps_svg, "Dashed line: required FPS.")
                  + _chart_card("Peak RAM by device", "", ram_svg, "Bars differ by device because each container has its own limits and thread count.")
                  + _chart_card("Detections per frame by device", "", det_svg,
                                "The same weights run on the same frames on the host, so these should be near-identical; the device only changes the estimated timing.")
                  + '</div>')
    return (f'{head}{warn}<div class="tables" style="margin-bottom:16px">{info}</div>{table}{_model_configs_card(ctx, L, model)}'
            f'{charts}{_problem_rows(ctx, model=model)}')


def _model_configs_card(ctx: Ctx, L: Links, model: str) -> str:
    """Per-configuration table of a model: quality (device-independent) and the estimated FPS on every device."""
    reports = {t: ctx.report(model, t) for t in ctx.targets}
    reports = {t: r for t, r in reports.items() if r}
    if not reports:
        return ""
    first = next(iter(reports.values()))
    entries = sweep_html.sorted_configs(first["sweep"]["configs"])
    head_dev = "".join(f'<th class="num"><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a><br><span class="s">{_e(t)}</span></th>' for t in reports)
    rows = []
    for e in entries:
        ag, r = e.get("agreement"), e.get("ratings") or {}
        cells = []
        for t, rep in reports.items():
            ent = next((x for x in rep["sweep"]["configs"] if x["id"] == e["id"]), None)
            if ent is None:
                cells.append('<td class="hm h-na">n/a</td>')
                continue
            is_best = rep["sweep"]["best"]["config_id"] == e["id"]
            cls = "green" if ent["realtime"]["realtime_capable"] else "red"
            cells.append(f'<td class="hm h-{cls}" title="{"best configuration on this device" if is_best else "est. FPS on this device"}">'
                         f'{_e(_f(sweep_mod.entry_fps(ent), 1))}{" &#9733;" if is_best else ""}</td>')
        rows.append(
            f'<tr><td>{_e(e["source"]["text"])}</td><td>{_e(e["input"]["label"])}{" (locked)" if e["input"].get("locked") else ""}</td>'
            f'<td class="num">{_e(sweep_mod.conf_text(e["conf"]))}</td><td class="num">{_e(_f(e["detections"]["mean_per_frame"], 2))}</td>'
            f'<td class="num">{_e(_f(ag["f1"], 3)) if ag else "n/a"}</td><td class="num">{_e(_f(e.get("temporal"), 3))}</td>'
            f'<td class="num"><b>{_e(_f(e.get("stability"), 3))}</b></td><td class="num">{_e(_f(r.get("coverage")))}</td>'
            f'<td class="num">{_e(_f(r.get("duplicates")))}</td>{"".join(cells)}</tr>')
    return ('<div class="card"><h2>All configurations of this model</h2><div class="scroll"><table class="cfgtable"><tr><th>Source resolution</th>'
            '<th>Input size</th><th class="num">Conf</th><th class="num">Dets/frame</th><th class="num">Agreement F1</th><th class="num">Temporal</th>'
            f'<th class="num">Stability</th><th class="num">Coverage (1-5)</th><th class="num">Duplicates</th>{head_dev}</tr>{"".join(rows)}</table></div>'
            '<p class="note">Quality columns do not depend on the device (the same weights run on the same frames); the device columns are the estimated FPS '
            '(host-measured for non-simulated devices), green = real-time, a star marks the best configuration of that device. Coverage and duplicates are '
            'your manual ratings: enter them in the web app (&#11088; Rate detection quality on the suite card) or with scripts/rate_configs.py, then '
            'rebuild the report.</p></div>')


def method_body(ctx: Ctx, L: Links) -> str:
    m = ctx.m
    anchors = "".join(
        f'<tr><td><a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a></td><td>{_e(t)}</td>'
        + (f'<td>{_e(ti["calibration"]["model"])}</td><td class="num">{_e(ti["calibration"]["gflops"])}</td>'
           f'<td class="num">{_e(ti["calibration"]["latency_ms"])}</td><td class="num">{_e(_f(ti.get("min_inference_ms")))}</td>'
           f'<td class="num">{_e(_f(ti.get("cpu_scale"), 3))}</td><td>{_link_text(ti["calibration"].get("source"))}</td></tr>'
           if ti.get("calibration") else '<td colspan="6" class="s">not simulated: host-measured</td></tr>')
        for t in ctx.targets for ti in [ctx.ti.get(t) or {}])
    limits = "".join(
        f'<tr><td>{_e(t)}</td><td class="num">{_e(_f(((ctx.ti.get(t) or {}).get("container") or {}).get("cpu_limit_cores")))}</td>'
        f'<td class="num">{_e(_f(((ctx.ti.get(t) or {}).get("container") or {}).get("memory_limit_mb"), 0))}</td>'
        f'<td class="num">{_e(_f((ctx.ti.get(t) or {}).get("ram_limit_mb"), 0))}</td>'
        f'<td class="num">{_e(_f((ctx.ti.get(t) or {}).get("cpu_cores")))}</td></tr>' for t in ctx.targets)
    env = m.get("environment") or {}
    host = m.get("host") or {}
    env_kv = _kv("Environment of the runs", [
        ("Simulation host (targets.yaml)", f"{host.get('name', 'n/a')}, single-core score {host.get('cpu_single_core_score', 'n/a')}"),
        ("Execution", ctx.cfg.get("execution")), ("CPU seen by the runs", env.get("cpu_model") or "n/a"),
        ("Platform", env.get("platform") or "n/a"), ("Python", env.get("python") or "n/a"),
        ("Packages", ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items() if v) or "n/a"),
        ("Frames per run / warmup", f"{ctx.cfg['frames']} / {ctx.cfg['warmup']}"),
        ("Source", Path(ctx.cfg["video"]).name if ctx.cfg.get("video") else "synthetic random frames (detection counts are not meaningful)"),
        ("Confidence threshold", ctx.cfg.get("conf") if ctx.cfg.get("conf") is not None else "from configs/detection.yaml"),
        ("Sweep", (f"source heights {ctx.cfg['sweep']['source_heights']} (0 = native), input sizes {ctx.cfg['sweep']['input_sizes'] or 'model default'}, "
                   f"confidence {ctx.cfg['sweep']['conf_thresholds']}") if ctx.cfg.get("sweep") else "off: one configuration per pair"),
        ("Required FPS", f"{ctx.required:g}"),
    ])
    return f"""<div class="head"><div><h1>Methodology</h1><div class="sub">How the numbers of this suite were produced</div></div></div>
<div class="card"><h2>Method</h2><p>{_e(METHOD_NOTE)}</p>
<ol class="tight">
<li><b>Functional run.</b> Every model runs for real on the host, inside a Docker service that is limited to the device's CPU cores and RAM
(<code>docker/docker-compose.yml</code>), with the runtime thread count set to the device's core count. Detections, peak RAM and CPU-side behaviour are measured.</li>
<li><b>Timing estimate.</b> Per frame: <code>inference_ms = max(min_inference_ms, anchor_latency_ms x model_GFLOPs / anchor_GFLOPs)</code>, where the anchor is a published
benchmark of a YOLO-class model on that exact device (table below).</li>
<li><b>CPU-side work</b> (resize, normalise, NMS) is measured on the host and scaled by <code>host_score / device_score</code> (single-core CPU scores).</li>
<li><b>Estimated FPS</b> = 1000 / mean estimated latency per frame. The host-measured x86-cpu target is not simulated: its FPS is the measured host throughput.</li>
<li>Benchmarks run sequentially, one model at a time in its own process, so peak RAM and timings of one run do not pollute the next.
Video decoding is timed separately and excluded from latency and FPS.</li>
</ol></div>
<div class="card"><h2>Calibration anchors</h2><div class="scroll"><table><tr><th>Device</th><th>Target</th><th>Anchor model</th><th class="num">GFLOPs</th>
<th class="num">Anchor latency ms</th><th class="num">Min inference ms</th><th class="num">CPU scale</th><th>Source</th></tr>{anchors}</table></div></div>
<div class="card"><h2>Container limits</h2><div class="scroll"><table><tr><th>Target</th><th class="num">cgroup CPU cores (measured)</th><th class="num">cgroup memory MB (measured)</th>
<th class="num">RAM budget MB (targets.yaml)</th><th class="num">CPU cores modelled</th></tr>{limits}</table></div></div>
<div class="tables">{env_kv}
<div class="card"><h2>Colour and verdict rules</h2><ul class="tight">
<li><b>Required FPS</b>: <code>benchmark.required_fps</code> in configs/detection.yaml if set, else the video's own FPS, else 25.</li>
<li><b>FPS heatmap</b>: green at or above the required FPS, amber from {AMBER_FPS:g} FPS, red below.</li>
<li><b>RAM heatmap</b>: peak RAM as % of the device budget; amber from {RAM_WARN_PCT:g}%, red from {RAM_FAIL_PCT:g}% (verdict warns when headroom is under 10%).</li>
<li><b>Verdict per run</b>: OK when real-time capable with P95 inside the frame budget and RAM fits; WARN when average speed is fine but P95 is late,
the device reaches at least half the required FPS, headroom is thin, or the output format is not decoded; FAIL below half real-time or when the RAM budget is exceeded.</li>
<li><b>Output format</b>: only YOLOv8/11 and end-to-end detection heads are decoded. Other models (face, pose, embedding) are timed with real inference but report 0 detections.</li>
<li><b>Failed</b> = the run crashed or the model cannot be loaded (for example missing external weight data). <b>Skipped</b> = device-native formats
(.engine, .rknn, .hef) that need the vendor runtime on real hardware.</li></ul></div></div>
{sweep_html.rules_card() if ctx.has_sweep else ""}
<div class="card"><h2>What is and is not modelled</h2><p class="note" style="font-size:.85rem">{_e(ACCURACY_DISCLAIMER)}</p>
<p class="note" style="font-size:.85rem">Container limits: the compose services are configured with the same core count and RAM as the device, but the CPU itself is the
host's: only the CPU-side work is scaled analytically. Throttling, memory bandwidth and accelerator operator coverage are not simulated.</p></div>
"""


# ---------------------------------------------------------------- page shell

def _nav(ctx: Ctx, prefix: str, active: str) -> str:
    L = Links(ctx, prefix=prefix)
    dev_menu = "".join(f'<a href="{_e(L.device(t))}">{_e(ctx.device(t))}</a>' for t in ctx.targets)
    mod_menu = "".join(f'<a href="{_e(L.model(m))}">{_e(short_model(m))}</a>' for m in ctx.models)
    a = lambda key: ' class="active"' if active == key else ""
    return (f'<nav class="topnav"><span class="brand">Benchmark suite</span>'
            f'<a href="{prefix}index.html"{a("overview")}>Overview</a>'
            f'<details class="dd"><summary{a("device")}>Devices &#9662;</summary><div class="menu">{dev_menu}</div></details>'
            f'<details class="dd"><summary{a("model")}>Models &#9662;</summary><div class="menu">{mod_menu}</div></details>'
            f'<a href="{prefix}method.html"{a("method")}>Methodology</a><span class="sp"></span>'
            f'<a href="{prefix}print.html"{a("print")}>Print all</a></nav>')


def _pager(prev: Optional[Tuple[str, str]], nxt: Optional[Tuple[str, str]]) -> str:
    if not prev and not nxt:
        return ""
    left = f'<a href="{_e(prev[0])}">&larr; {_e(prev[1])}</a>' if prev else "<span></span>"
    right = f'<a href="{_e(nxt[0])}">{_e(nxt[1])} &rarr;</a>' if nxt else "<span></span>"
    return f'<div class="pager">{left}{right}</div>'


def _shell(ctx: Ctx, title: str, body: str, prefix: str, active: str, crumbs: List[Tuple[Optional[str], str]],
           pager: str = "") -> str:
    crumb_html = " &rsaquo; ".join(f'<a href="{_e(h)}">{_e(t)}</a>' if h else _e(t) for h, t in crumbs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)} | {_e(ctx.m['suite_id'])}</title>
<style>{CSS}{SITE_CSS}</style>
</head>
<body>
<div class="wrap">
{_nav(ctx, prefix, active)}
<div class="crumbs">{crumb_html}</div>
{body}
{pager}
<div class="foot">Suite {_e(ctx.m['suite_id'])}, generated by Model Inference Benchmark. Device figures are estimates (roughly +/-30-50% versus real hardware).
This page is self-contained: use the browser's Print to save it as PDF, or open <a href="{prefix}print.html">Print all</a> for every page in one document.</div>
</div>
</body>
</html>
"""


def _siblings(items: List[str], i: int, href, label) -> str:
    prev = (href(items[i - 1]), label(items[i - 1])) if i > 0 else None
    nxt = (href(items[i + 1]), label(items[i + 1])) if i + 1 < len(items) else None
    return _pager(prev, nxt)


def render_pages(ctx: Ctx) -> Dict[str, str]:
    """{relative path: html} for every page of the site."""
    pages: Dict[str, str] = {}
    root, sub = Links(ctx), Links(ctx, prefix="../")
    pages["index.html"] = _shell(ctx, "Overview", overview_body(ctx, root), "", "overview", [(None, "Overview")])
    for i, t in enumerate(ctx.targets):
        pages[f"devices/{ctx.dev_slug[t]}.html"] = _shell(
            ctx, ctx.device(t), device_body(ctx, sub, t), "../", "device",
            [("../index.html", "Overview"), (None, f"Device: {ctx.device(t)}")],
            _siblings(ctx.targets, i, lambda x: f"{ctx.dev_slug[x]}.html", ctx.device))
    for i, mod in enumerate(ctx.models):
        pages[f"models/{ctx.model_slug[mod]}.html"] = _shell(
            ctx, short_model(mod), model_body(ctx, sub, mod), "../", "model",
            [("../index.html", "Overview"), (None, f"Model: {short_model(mod)}")],
            _siblings(ctx.models, i, lambda x: f"{ctx.model_slug[x]}.html", short_model))
    pages["method.html"] = _shell(ctx, "Methodology", method_body(ctx, root), "", "method",
                                  [("index.html", "Overview"), (None, "Methodology")])

    # print.html: every page in one document, links become in-document anchors
    P = Links(ctx, print_mode=True)
    toc = ('<div class="card noprint toc"><h2>Contents</h2><a href="#overview">Overview</a>'
           + "".join(f'<a href="#device-{ctx.dev_slug[t]}">Device: {_e(ctx.device(t))}</a>' for t in ctx.targets)
           + "".join(f'<a href="#model-{ctx.model_slug[m]}">Model: {_e(short_model(m))}</a>' for m in ctx.models)
           + '<a href="#method">Methodology</a></div>')
    sections = [f'<section id="overview" class="printpage">{overview_body(ctx, P)}</section>']
    sections += [f'<section id="device-{ctx.dev_slug[t]}" class="printpage">{device_body(ctx, P, t)}</section>' for t in ctx.targets]
    sections += [f'<section id="model-{ctx.model_slug[m]}" class="printpage">{model_body(ctx, P, m)}</section>' for m in ctx.models]
    sections.append(f'<section id="method" class="printpage">{method_body(ctx, P)}</section>')
    pages["print.html"] = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark report (all pages) | {_e(ctx.m['suite_id'])}</title>
<style>{CSS}{SITE_CSS}</style>
</head>
<body>
<div class="wrap">
<nav class="topnav noprint"><span class="brand">Benchmark suite: all pages</span><a href="index.html">Back to the site</a><span class="sp"></span>
<span class="s">Use the browser's Print, then Save as PDF (each page starts on a new sheet). Per-run reports are linked, not included.</span></nav>
{toc}
{"".join(sections)}
</div>
</body>
</html>
"""
    return pages


def write_site(folder: Path, manifest: Dict[str, Any]) -> None:
    ctx = Ctx(manifest, folder)
    (folder / "devices").mkdir(exist_ok=True)
    (folder / "models").mkdir(exist_ok=True)
    for rel, html in render_pages(ctx).items():
        _atomic_write(folder / rel, html)
