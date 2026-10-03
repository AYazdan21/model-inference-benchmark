"""
HTML pieces of a sweep: best-configuration banner, per-configuration table, charts, sample frames per configuration and the
selection rule. Used by the per-run report (html_report.py) and by the benchmark suite site (suite_site.py).
No scripts, inline SVG only; every dynamic value is escaped.
"""
from typing import Any, Dict, List, Optional, Sequence

from src.benchmark import sweep as sw
from src.benchmark.html_report import _chart_card, _e, _f, _legend, line_chart, stacked_bars

SERIES_CSS = ["s1", "s2", "s3", "s4", "s5", "s6"]


def sorted_configs(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Source resolution (highest first), input size (largest first), confidence (ascending)."""
    return sorted(entries, key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"]))


def _rating_cell(entry: Dict[str, Any], key: str) -> str:
    r = entry.get("ratings") or {}
    v = r.get(key)
    return f'<td class="num">{_e(v)}</td>' if v is not None else '<td class="num s">&ndash;</td>'


def best_banner(report: Dict[str, Any], rate_hint: bool = True) -> str:
    sw_ = report["sweep"]
    best = sw_["best"]
    entry = sw.best_entry(report)
    dims = sw_["dimensions"]
    n_src, n_in, n_conf = len(dims["source"]), len(dims["input_sizes"]), len(dims["conf_thresholds"])
    rated = [e for e in sw_["configs"] if e.get("ratings")]
    stamps = sorted(str(e["ratings"].get("updated") or "") for e in rated)
    parts = [
        '<div class="card best-banner">',
        '<div class="l">BEST CONFIGURATION</div>',
        f'<div class="big">{_e(entry["label"])}</div>',
        f'<div><b>Selected by:</b> {_e(sw.RULE_NAMES.get(best["rule"], best["rule"]))}</div>',
        f'<div><b>Why:</b> {_e(best["reason"])}</div>',
    ]
    if best.get("auto_pick"):
        ap = best["auto_pick"]
        parts.append(f'<div class="s"><b>The automatic rule alone would pick:</b> {_e(ap["label"])} &mdash; {_e(ap["reason"])}</div>')
    if rated:
        last = stamps[-1].replace("T", " ") if stamps and stamps[-1] else "n/a"
        parts.append(f'<div class="s">Manual ratings used: {len(rated)} of {len(sw_["configs"])} configurations rated for {_e(sw_.get("video"))} '
                     f'(last change {_e(last)}).</div>')
    elif rate_hint:
        parts.append('<div class="s">No manual ratings yet, so the automatic real-time + stable rule decides. Rate the detection quality of the '
                     'configurations in the web app (&#11088; Rate detection quality) or with <code>scripts/rate_configs.py</code>; '
                     'ratings take priority and are applied to every report of the same video and model.</div>')
    lock = (f' The model input is locked to {_e(dims["input_sizes"][0][0])} x {_e(dims["input_sizes"][0][1])}: only the source '
            'resolution and the confidence were swept.') if dims.get("input_locked") else ""
    parts.append(f'<div class="s">Tested {len(sw_["configs"])} configurations: {n_src} source resolution(s) x {n_in} input size(s) x '
                 f'{n_conf} confidence value(s).{lock} Reference for the agreement score: {_e(sw.find_entry(report, sw_["reference_config"])["label"])}.</div>')
    parts.append("</div>")
    return "".join(parts)


def _fps_head(simulated: bool) -> str:
    return "Est. FPS" if simulated else "Host FPS"


def config_table(report: Dict[str, Any]) -> str:
    sw_ = report["sweep"]
    simulated = report["device_estimate"] is not None
    best_id = sw_["best"]["config_id"]
    rows = []
    for e in sorted_configs(sw_["configs"]):
        is_best = e["id"] == best_id
        fps = sw.entry_fps(e)
        rt = e["realtime"]["realtime_capable"]
        ag, tc, st = e.get("agreement"), e.get("temporal"), e.get("stability")
        tags = (' <span class="tag ref" title="reference configuration: agreement is measured against it">ref</span>' if e["is_reference"] else "")
        lat = (e["device_estimate"]["latency_ms"]["p95"] if e["device_estimate"] else e["host_performance"]["latency_ms"]["p95"])
        rows.append(
            f'<tr class="{"best" if is_best else ""}"><td>{"&#9733; " if is_best else ""}{_e(e["source"]["text"])}</td>'
            f'<td>{_e(e["input"]["label"])}</td><td class="num">{_e(sw.conf_text(e["conf"]))}</td>'
            f'<td class="num"><b>{_e(_f(fps, 1))}</b></td><td class="{"rt-yes" if rt else "rt-no"}">{"&#10003;" if rt else "&#10007;"}</td>'
            f'<td class="num">{_e(_f(lat, 1))}</td><td class="num">{_e(_f(e["host_performance"]["throughput_fps"], 1))}</td>'
            f'<td class="num">{_e(_f(e.get("gflops"), 2))}</td><td class="num">{_e(_f(e["resources"]["peak_ram_mb"], 0))}</td>'
            f'<td class="num">{_e(_f(e["detections"]["mean_per_frame"], 2))}</td>'
            f'<td class="num">{_e(_f(ag["f1"], 3)) if ag else "n/a"}{tags}</td><td class="num">{_e(_f(tc, 3)) if tc is not None else "n/a"}</td>'
            f'<td class="num"><b>{_e(_f(st, 3)) if st is not None else "n/a"}</b></td>'
            f'{_rating_cell(e, "coverage")}{_rating_cell(e, "duplicates")}</tr>')
    head = (f'<tr><th>Source resolution</th><th>Input size</th><th class="num">Conf</th><th class="num">{_fps_head(simulated)}</th>'
            '<th>Real-time</th><th class="num">P95 ms</th><th class="num">Host FPS</th><th class="num">GFLOPs</th>'
            '<th class="num">Peak RAM MB</th><th class="num">Dets/frame</th><th class="num">Agreement F1</th>'
            '<th class="num">Temporal</th><th class="num">Stability</th><th class="num">Coverage (1-5)</th><th class="num">Duplicates</th></tr>')
    note = (f'FPS and latency are {"estimated on the device" if simulated else "measured on the host"}; they do not depend on the confidence. '
            f'Required FPS: {report["realtime"]["required_fps"]:g}. Agreement F1, temporal consistency and stability are label-free '
            'proxies (see the selection rule below); coverage and duplicates are your manual ratings. The highlighted row is the best configuration.')
    return (f'<div class="card"><h2>All configurations ({len(sw_["configs"])})</h2><div style="overflow-x:auto">'
            f'<table class="cfgtable">{head}{"".join(rows)}</table></div><p class="note">{_e(note)}</p></div>')


def _series_by_input(entries: Sequence[Dict[str, Any]], conf: float):
    out = {}
    for e in entries:
        if abs(e["conf"] - conf) < 1e-9:
            out.setdefault(e["input"]["label"], []).append(e)
    return out


def sweep_charts(report: Dict[str, Any]) -> str:
    sw_ = report["sweep"]
    entries = sw_["configs"]
    dims = sw_["dimensions"]
    simulated = report["device_estimate"] is not None
    default_conf = dims["default_conf"]
    best = sw.best_entry(report)
    cards = []

    # FPS vs source resolution, one line per input size
    by_input = _series_by_input(entries, default_conf)
    series, legend = [], []
    for i, (lab, es) in enumerate(sorted(by_input.items(), key=lambda kv: -int(kv[0].split("x")[0]))):
        es = sorted(es, key=lambda e: e["source"]["height"])
        css = SERIES_CSS[i % len(SERIES_CSS)]
        series.append((css, [float(e["source"]["height"]) for e in es], [sw.entry_fps(e) for e in es]))
        legend.append((f"input {lab}", css))
    svg = line_chart(series, "Source frame height (px)", "Estimated FPS" if simulated else "Host FPS",
                     report["realtime"]["required_fps"], "FPS versus source resolution")
    cards.append(_chart_card("FPS versus source resolution", _legend(legend), svg,
                             f'One line per model input size. Dashed red line: required {report["realtime"]["required_fps"]:g} FPS.'))

    # latency breakdown per (source, input)
    rows = []
    for e in sorted((e for e in entries if abs(e["conf"] - default_conf) < 1e-9), key=lambda e: (-e["source"]["height"], -e["input"]["h"])):
        hp, est = e["host_performance"], e["device_estimate"]
        if est:
            segs = [("CPU-side (pre+post)", est["cpu_side_ms"] or 0.0, "pre"), ("Inference (device est.)", est["inference_ms"] or 0.0, "dev")]
        else:
            st = hp["stages"]
            segs = [("Pre+post", ((st["preprocess"] or {}).get("mean") or 0) + ((st["postprocess"] or {}).get("mean") or 0), "pre"),
                    ("Inference (host)", (st["inference"] or {}).get("mean") or 0.0, "host")]
        rows.append((f'{e["source"]["label"]} · {e["input"]["h"] if e["input"]["h"] == e["input"]["w"] else e["input"]["label"]}', segs))
    cards.append(_chart_card("Latency breakdown per configuration",
                             _legend([("CPU-side (pre + post-processing)", "pre"), ("Inference (device est.)" if simulated else "Inference (host)", "dev" if simulated else "host")]),
                             stacked_bars(rows, "Latency breakdown per configuration", label_w=150, label_chars=22),
                             "Mean per frame, independent of the confidence. Postprocess time was measured at the lowest confidence."))

    # detections / stability versus confidence, one line per source resolution at the best configuration's input size
    in_label = best["input"]["label"]
    series_d, series_s, legend_s = [], [], []
    for i, src in enumerate(sorted(dims["source"], key=lambda s: -s["height"])):
        es = sorted((e for e in entries if e["source"]["label"] == src["label"] and e["input"]["label"] == in_label), key=lambda e: e["conf"])
        css = SERIES_CSS[i % len(SERIES_CSS)]
        legend_s.append((src["text"], css))
        series_d.append((css, [e["conf"] * 100 for e in es], [float(e["detections"]["mean_per_frame"]) for e in es]))
        pts = [(e["conf"] * 100, e["stability"] * 100) for e in es if e.get("stability") is not None]
        if pts:
            series_s.append((css, [p[0] for p in pts], [p[1] for p in pts]))
    scope = f"Lines: source resolutions, model input {in_label}."
    cards.append(_chart_card("Detections per frame versus confidence", _legend(legend_s),
                             line_chart(series_d, "Confidence threshold (%)", "Detections per frame", None, "Detections versus confidence"), scope))
    if series_s:
        cards.append(_chart_card("Stability versus confidence", _legend(legend_s),
                                 line_chart(series_s, "Confidence threshold (%)", "Stability (%)", None, "Stability versus confidence", ymax=100.0),
                                 scope + " Agreement is measured against the reference configuration, so it peaks near the run's confidence."))
    else:
        cards.append(_chart_card("Stability versus confidence", "", line_chart([], "", "", None, "Stability"),
                                 "Not measurable: the output format of this model is not decoded by the harness."))
    return '<div class="charts">' + "".join(cards) + "</div>"


def sample_grid(report: Dict[str, Any]) -> str:
    sw_ = report["sweep"]
    frames = sw_.get("sample_frames") or []
    blocks = []
    for e in sorted_configs(sw_["configs"]):
        if not e.get("samples"):
            continue
        figs = "".join(f'<figure><img alt="{_e(e["short"])}, frame {_e(rel)}" src="{_e(rel)}" loading="lazy"><figcaption>Frame {_e(rel.rsplit("_f", 1)[-1][:-4].lstrip("0") or "0")}</figcaption></figure>'
                       for rel in e["samples"])
        star = " &#9733; best" if e["id"] == sw_["best"]["config_id"] else ""
        blocks.append(f'<details class="cfgsamples"><summary>{_e(e["label"])}{star}</summary><div class="samples">{figs}</div></details>')
    if not blocks:
        return ""
    return (f'<div class="card"><h2>Sample frames of every configuration</h2><p class="note">The same frames ({_e(", ".join(str(i) for i in frames))}; '
            'the ones with the most detections in the reference configuration) are shown for every configuration, annotated with that '
            'configuration\'s boxes and confidence values, so they can be compared side by side. The images are in the <code>samples/</code> folder next to '
            'this file.</p>' + "".join(blocks) + "</div>")


def rules_card() -> str:
    steps = "".join(f"<li>{_e(s)}</li>" for s in sw.SELECTION_RULE_STEPS)
    notes = "".join(f"<li>{_e(s)}</li>" for s in sw.SWEEP_NOTES)
    return (f'<div class="card"><h2>How the best configuration is chosen</h2><ol class="tight">{steps}</ol>'
            f'<p class="note">{_e(sw.STABILITY_FORMULA)}. Agreement F1 is computed per frame by matching boxes of the same class with IoU &ge; {sw.IOU_AGREEMENT}; '
            f'temporal consistency matches boxes of consecutive frames with IoU &ge; {sw.IOU_TEMPORAL}.</p>'
            f'<h2 style="margin-top:12px">Method notes</h2><ul class="tight">{notes}</ul></div>')


def best_line_text(best_cell: Optional[Dict[str, Any]]) -> str:
    """One-line description of the best configuration of a pair, e.g. for lists: '720p · 640 · 0.35 (manual rating)'."""
    if not best_cell:
        return ""
    return f'{best_cell["short"]} ({sw.RULE_NAMES.get(best_cell["rule"], best_cell["rule"])})'
