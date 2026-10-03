"""Writes a benchmark report bundle: report.json, frames.csv, summary.md, samples/*.jpg and report.html."""
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from src.benchmark import sweep as sw
from src.benchmark.html_report import render_html
from src.utils.logger import get_logger
from src.video.annotator import VideoAnnotator

logger = get_logger("ReportExport")

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "results" / "reports"
CSV_COLUMNS = [
    "frame_index", "decode_ms", "host_latency_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
    "sim_latency_ms", "sim_inference_ms", "detections", "class_counts", "max_confidence", "rss_mb", "cpu_percent",
]
SWEEP_CSV_COLUMNS = ["config_id", "source", "input", "conf"] + CSV_COLUMNS
_NUM_COLUMNS = ("decode_ms", "host_latency_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "sim_latency_ms",
                "sim_inference_ms", "max_confidence", "rss_mb", "cpu_percent")


def render_samples(top_frames: List[Dict[str, Any]], report: Dict[str, Any]) -> List[Tuple[int, bytes]]:
    """Annotates the kept frames (boxes + latency overlay) and encodes them as JPEG."""
    sim = report["device_estimate"]
    fps = sim["est_fps"] if sim else report["host_performance"]["throughput_fps"]
    annotator = VideoAnnotator(report["target"]["key"], report["target"]["device"] if sim else "")
    out = []
    for item in sorted(top_frames, key=lambda t: t["frame_index"]):
        img = annotator.annotate(item["frame"], item["result"], item["frame_index"], fps)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append((item["frame_index"], buf.tobytes()))
    return out


def _fmt_counts(counts: Dict[str, int]) -> str:
    return ";".join(f"{name}:{n}" for name, n in counts.items())


def write_frames_csv(path: Path, per_frame: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    columns = columns or CSV_COLUMNS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in per_frame:
            row = {k: ("" if rec.get(k) is None else rec[k]) for k in columns}
            row["class_counts"] = _fmt_counts(rec.get("class_counts") or {})
            writer.writerow(row)


def read_frames_csv(path: Path, config_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Per-frame records of a frames.csv (of one configuration for a sweep report), in the format the profiler produces."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if config_id is not None and row.get("config_id") != config_id:
                continue
            rec: Dict[str, Any] = {"frame_index": int(row["frame_index"]), "detections": int(row["detections"] or 0)}
            for k in _NUM_COLUMNS:
                rec[k] = float(row[k]) if row.get(k) not in (None, "") else None
            counts = {}
            for part in (row.get("class_counts") or "").split(";"):
                if ":" in part:
                    name, n = part.rsplit(":", 1)
                    counts[name] = int(n)
            rec["class_counts"] = counts
            out.append(rec)
    return out


def _notes_line(note: str) -> str:
    """'- Notes: <text>' list item (continuation lines indented so they stay inside the item). A '<' is
    backslash-escaped so a Markdown viewer that does not sanitise HTML cannot be made to run markup from the note."""
    text = (note or "").replace("<", "\\<").replace("\n", "\n  ")
    return f"- Notes: {text}" if text else "- Notes:"


def build_summary_md(report: Dict[str, Any]) -> str:
    t, m, cfg = report["target"], report["model"], report["config"]
    host, rt, res, dets = report["host_performance"], report["realtime"], report["resources"], report["detections"]
    est = report["device_estimate"]
    lines = [
        f"# Benchmark: {m['file']} on {t['device']} ({t['key']})",
        "",
        f"- Report: `{report['report_id']}` ({report['created']})",
        f"- Overall verdict: **{report['verdict']['overall'].upper()}**",
        _notes_line(report.get("notes", "")),
        f"- Source: {cfg['source'].get('file') or 'synthetic frames'}, {cfg['frames_processed']} frames "
        f"(warmup {cfg['warmup']}), conf {cfg['conf_threshold']}",
    ]
    if report.get("sweep"):
        best = report["sweep"]["best"]
        lines.append(f"- Best configuration: **{sw.best_entry(report)['label']}** (selected by: {sw.RULE_NAMES.get(best['rule'], best['rule'])})")
        lines.append("- Why: " + best["reason"].replace("<", "\\<"))
        if best.get("auto_pick"):
            lines.append(f"- Automatic pick without ratings: {best['auto_pick']['label']}")
        lines.append(f"- Configurations tested: {len(report['sweep']['configs'])} (all in the table below and in frames.csv)")
    lines += [
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    if est:
        lines += [
            f"| Est. device FPS | {est['est_fps']} |",
            f"| Est. device latency P50 / P95 | {est['latency_ms']['p50']} / {est['latency_ms']['p95']} ms |",
            f"| Est. device inference | {est['inference_ms']} ms |",
        ]
    lines += [
        f"| Host FPS (measured, processing only) | {host['throughput_fps']} |",
        f"| Host latency P50 / P95 | {host['latency_ms']['p50']} / {host['latency_ms']['p95']} ms |",
        f"| Real-time factor vs {rt['required_fps']:g} FPS | {rt['device_realtime_factor']}x "
        f"({'capable' if rt['realtime_capable'] else 'not real-time'}) |",
        f"| Detections per frame | {dets['mean_per_frame']} (total {dets['total']}, "
        f"{dets['frames_with_detections_pct']}% of frames) |",
        f"| Peak RAM / limit | {res['peak_ram_mb']} MB / {res['ram_limit_mb'] or 'none'} MB |",
        f"| Avg CPU of target cores | {res['avg_cpu_percent']} % |",
        f"| Model load / first inference | {report['cold_start']['model_load_ms']} / "
        f"{report['cold_start']['first_inference_ms']} ms |",
        f"| Model | {m['size_mb']} MB, {m['gflops']} GFLOPs, output format: {m['output_format']} |",
        "",
        "## Verdict",
        "",
    ]
    lines += [f"- **{i['level'].upper()}**: {i['message']}" for i in report["verdict"]["items"]]
    if report.get("sweep"):
        simulated = report["device_estimate"] is not None
        lines += ["", "## Configurations", "",
                  f"| Source | Input | Conf | {'Est. FPS' if simulated else 'Host FPS'} | Real-time | Agreement F1 | Temporal | Stability | Coverage | Duplicates | Best |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        best_id = report["sweep"]["best"]["config_id"]
        fmt = lambda v: "n/a" if v is None else v
        for e in sorted(report["sweep"]["configs"], key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"])):
            ag, r = e.get("agreement"), e.get("ratings") or {}
            lines.append(f"| {e['source']['text']} | {e['input']['label']} | {sw.conf_text(e['conf'])} | {sw.entry_fps(e):.1f} | "
                         f"{'yes' if e['realtime']['realtime_capable'] else 'no'} | {fmt(ag['f1'] if ag else None)} | {fmt(e.get('temporal'))} | "
                         f"{fmt(e.get('stability'))} | {fmt(r.get('coverage'))} | {fmt(r.get('duplicates'))} | {'*' if e['id'] == best_id else ''} |")
        lines += ["", sw.STABILITY_FORMULA + ". " + sw.SELECTION_RULE_STEPS[0]]
    lines += ["", "## Method and accuracy", "", t["method"], "", t["accuracy_disclaimer"], ""]
    return "\n".join(lines)


def write_report_bundle(
    report: Dict[str, Any],
    per_frame: List[Dict[str, Any]],
    samples: List[Tuple[int, bytes]],
    out_root: Optional[str | Path] = None,
    link_samples: bool = False,
    back_link: Optional[Tuple[str, str]] = None,
    sweep_files: Optional[Dict[str, bytes]] = None,
) -> Path:
    """Creates <out_root>/<report_id>/ and returns that folder.
    link_samples / back_link: see html_report.render_html (used for the per-run reports of a benchmark suite).
    Sweep reports: per_frame holds the records of every configuration (config_id column), samples the sample frames of the
    best configuration (for embedding) and sweep_files every sample image of every configuration by relative file name."""
    root = Path(out_root) if out_root else DEFAULT_REPORTS_DIR
    base_id = report["report_id"]
    n = 1
    while (root / report["report_id"]).exists():  # two runs inside the same second
        n += 1
        report["report_id"] = f"{base_id}-{n}"
    folder = root / report["report_id"]
    (folder / "samples").mkdir(parents=True)

    if report.get("sweep"):
        for rel, jpeg in (sweep_files or {}).items():
            (folder / rel).write_bytes(jpeg)
        report["render_opts"] = {"link_samples": link_samples, "back_link": list(back_link) if back_link else None}
        write_frames_csv(folder / "frames.csv", per_frame, SWEEP_CSV_COLUMNS)
        best_frames = [r for r in per_frame if r["config_id"] == report["sweep"]["best"]["config_id"]]
    else:
        report["samples"] = []
        for idx, jpeg in samples:
            rel = f"samples/frame_{idx:06d}.jpg"
            (folder / rel).write_bytes(jpeg)
            report["samples"].append(rel)
        write_frames_csv(folder / "frames.csv", per_frame)
        best_frames = per_frame
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "summary.md").write_text(build_summary_md(report), encoding="utf-8")
    (folder / "report.html").write_text(
        render_html(report, best_frames, samples, link_samples=link_samples, back_link=back_link), encoding="utf-8")
    logger.info(f"Report bundle written to: {folder}")
    return folder


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):  # Windows: replace fails while a reader has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def best_sample_bytes(folder: Path, report: Dict[str, Any]) -> List[Tuple[int, bytes]]:
    """(frame index, JPEG) of the sample images of the report's best configuration, read from the bundle folder."""
    out = []
    for rel in sw.best_entry(report).get("samples") or []:
        m = re.search(r"_f(\d+)\.jpg$", rel)
        f = folder / rel
        if m and f.is_file():
            out.append((int(m.group(1)), f.read_bytes()))
    return out


def rerender_bundle(folder: Path, store=None, force: bool = False) -> Optional[Dict[str, Any]]:
    """Applies the current manual ratings to a sweep report bundle and rewrites report.json, report.html and summary.md
    (frames.csv and the sample images do not change). Returns the report, or None for a report without sweep data.
    Nothing is rewritten when the ratings did not change anything, unless force is set."""
    path = folder / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("sweep"):
        return None
    changed = sw.apply_ratings(report, store=store)
    if not (changed or force):
        return report
    frames = read_frames_csv(folder / "frames.csv", sw.best_entry(report)["id"])
    opts = report.get("render_opts") or {}
    back = tuple(opts["back_link"]) if opts.get("back_link") else None
    _atomic_text(path, json.dumps(report, indent=2, ensure_ascii=False))
    _atomic_text(folder / "summary.md", build_summary_md(report))
    _atomic_text(folder / "report.html", render_html(report, frames, best_sample_bytes(folder, report),
                                                     link_samples=bool(opts.get("link_samples")), back_link=back))
    return report
