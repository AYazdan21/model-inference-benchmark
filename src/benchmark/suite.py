"""
Benchmark suite: every selected model on every selected target, with a manifest (suite.json) that is rewritten
after every cell so a partial, stopped or resumed suite can always be rendered.

Folder layout (results/reports/suite_<timestamp>/):
    suite.json      manifest: config, per-cell status, key metrics and user notes, timings
    runs/<id>/      one report bundle per finished cell (report.html, report.json, frames.csv, summary.md, samples/)
    raw/<target>.json   raw run summaries written by scripts/run_benchmark.py
    index.html, devices/, models/, method.html, print.html, results.csv, results.json   (see suite_site.py)

The runner starts one container per target and benchmarks its models one after another (never in parallel:
parallel runs would corrupt the timings). It prints machine-readable lines for the web app:
    SUITE_START <suite_id> <cells to run>
    SUITE_TARGET <target>
    SUITE_CONTAINER <docker container name>
    SUITE_CURRENT <target> <model>
    SUITE_CELL <done>/<total> <target> <model> <status> [fps=<f> heat=<green|amber|red> dur=<seconds>]
    SUITE_DONE <suite_id>
plus the harness' own PROGRESS <done>/<total> and SWEEP_PASS <i>/<n> <source> <input> lines (sweep position).

Sweeps: every cell (model x device) is benchmarked over source resolutions x model input sizes x confidence thresholds
(config["sweep"]; a suite without it is the old single-configuration kind and stays that way when resumed). The cell keeps the
key metrics of the best configuration plus cell["sweep"] (which one, why); the per-run report.json holds all configurations.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.benchmark import sweep as sweep_mod
from src.benchmark.notes import clean_note, note_for
from src.utils.docker_runner import build_docker_run_command
from src.utils.logger import get_logger

logger = get_logger("BenchmarkSuite")

SUITE_ID_RE = re.compile(r"^suite_[0-9_]+$")
RUNNABLE_EXTS = {".onnx", ".pt"}
NATIVE_EXTS = {".engine", ".trt", ".rknn", ".hef"}
NATIVE_REASON = "device-native format, needs the vendor runtime on real hardware"
AMBER_FPS = 5.0            # below this a cell is red; from here up to the required FPS it is amber
DONE_STATUSES = {"ok", "skipped"}
RAM_WARN_PCT, RAM_FAIL_PCT = 75.0, 90.0

REPORT_LINE_RE = re.compile(r"\] REPORT ([A-Za-z0-9_.-]+)\s*$")
MODEL_LINE_RE = re.compile(r"=====\s*\[(\d+)/(\d+)\]\s*(.+?)\s*=====")
FAIL_LINE_RE = re.compile(r"Benchmark failed for (.+?): (.*)$")
PROGRESS_RE = re.compile(r"^PROGRESS \d+/\d+\s*$")
SWEEP_PASS_RE = re.compile(r"^SWEEP_PASS \d+/\d+ \S+ \S+\s*$")
ERROR_HINT_RE = re.compile(r"error|exception|traceback|killed|no such|cannot|denied", re.I)

# Rough duration model, fitted on one full run (14 models x 5 devices, 3 frames + 1 warmup, i7-8550U laptop, 572 s):
# container start per target, process start + report writing per model, model load per MB of weights, and host
# milliseconds per frame from GFLOPs. Good to maybe +/-30%; the web app shows it labelled as a rough estimate.
CONTAINER_START_S = 8.0
CELL_OVERHEAD_S = 4.0
LOAD_S_PER_MB = 0.08
HOST_MS_BASE, HOST_MS_PER_GFLOP = 40.0, 15.0
DEFAULT_GFLOPS = 10.0
# Sweeps: every (source resolution, input size) pass decodes its frames again (the video is decoded at native size, then
# downscaled) and runs them; the host time of a model grows with the input area. Input size 640 is the unit of HOST_MS_*.
DECODE_MS_PER_FRAME = 30.0
PASS_OVERHEAD_S = 1.0
EXTRA_WARMUP = 3
TORCH_START_S = 6.0   # a .pt model also imports PyTorch / Ultralytics and profiles its GFLOPs once per input size

CSV_COLUMNS = [
    "model", "target", "device", "status", "error", "report_id", "report_path", "config_id", "source_resolution", "input_size",
    "conf", "is_best", "selection_rule", "selection_reason", "agreement_f1", "temporal", "stability", "coverage", "duplicates",
    "n_configs", "simulated", "fps", "est_fps",
    "est_latency_p50_ms", "est_latency_p95_ms", "host_fps", "host_latency_p50_ms", "host_latency_p95_ms",
    "cpu_side_ms", "inference_ms", "peak_ram_mb", "ram_limit_mb", "ram_pct", "dets_per_frame",
    "frames_with_detections_pct", "cold_start_ms", "first_inference_ms", "output_format", "required_fps",
    "realtime_capable", "verdict", "duration_s", "notes",
]


# ---------------------------------------------------------------- selection and estimates

def pick_models(requested: List[str], models_dir: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """(runnable model file names, skipped native-format models with reasons) for 'all' or explicit paths/names."""
    if requested == ["all"]:
        found = sorted(p for p in models_dir.glob("*") if p.suffix.lower() in RUNNABLE_EXTS | NATIVE_EXTS)
    else:
        found = [Path(n) for n in requested]
    runnable = [p.name for p in found if p.suffix.lower() in RUNNABLE_EXTS]
    skipped = [{"model": p.name, "reason": NATIVE_REASON} for p in found if p.suffix.lower() in NATIVE_EXTS]
    return runnable, skipped


def sweep_passes(sweep: Optional[Dict[str, Any]], dynamic: bool, default_size: int = 640) -> Tuple[int, float]:
    """(number of passes, mean relative input area) of one model for the sweep settings: passes are source resolutions x
    input sizes (one size for a model with a fixed input); inference cost scales with the input area."""
    if not sweep or sweep.get("enabled") is False:
        return 1, 1.0
    heights = max(1, len(sweep.get("source_heights") or [0]))
    sizes = sorted(set(sweep.get("input_sizes") or []) | {default_size}) if dynamic else [default_size]
    area = sum((z / default_size) ** 2 for z in sizes) / len(sizes)
    return heights * len(sizes), area


def estimate_seconds(models: List[Tuple], n_targets: int, frames: int, warmup: int,
                     sweep: Optional[Dict[str, Any]] = None) -> float:
    """Rough suite duration from [(GFLOPs or None, weights size in MB[, dynamic input[, is a .pt model]])] per model: not a promise.
    Confidence values cost nothing (derived by filtering); source resolutions and input sizes cost one pass each."""
    per_target = CONTAINER_START_S
    for item in models:
        gflops, size_mb = item[0], item[1]
        dynamic = bool(item[2]) if len(item) > 2 else False
        if len(item) > 3 and item[3]:
            per_target += TORCH_START_S
        passes, area = sweep_passes(sweep, dynamic)
        host_ms = HOST_MS_BASE + HOST_MS_PER_GFLOP * (gflops if gflops else DEFAULT_GFLOPS) * area
        if passes == 1 and not (sweep and sweep.get("enabled") is not False):
            per_target += CELL_OVERHEAD_S + LOAD_S_PER_MB * size_mb + (frames + warmup) * host_ms / 1000.0
        else:
            per_target += (CELL_OVERHEAD_S + LOAD_S_PER_MB * size_mb + warmup * host_ms / 1000.0
                           + passes * (frames * (host_ms + DECODE_MS_PER_FRAME) / 1000.0 + PASS_OVERHEAD_S + EXTRA_WARMUP * host_ms / 1000.0))
    return per_target * n_targets


def cached_gflops(project_root: Path) -> Dict[str, Optional[float]]:
    """GFLOPs per model from the last 'Estimate' run (results/metrics/_estimates.json), if any."""
    try:
        data = json.loads((project_root / "results" / "metrics" / "_estimates.json").read_text(encoding="utf-8"))
        return {r["name"]: r.get("gflops") for r in data.get("rows", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def cached_dynamic(project_root: Path) -> Dict[str, Optional[bool]]:
    """Whether each model accepts other input sizes, from the last 'Estimate' run (.pt weights are always dynamic)."""
    try:
        data = json.loads((project_root / "results" / "metrics" / "_estimates.json").read_text(encoding="utf-8"))
        return {r["name"]: r.get("dynamic_input") for r in data.get("rows", []) if r.get("dynamic_input") is not None}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


# ---------------------------------------------------------------- manifest

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def manifest_path(folder: Path) -> Path:
    return folder / "suite.json"


def load_manifest(folder: Path) -> Dict[str, Any]:
    return json.loads(manifest_path(folder).read_text(encoding="utf-8"))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):  # Windows: replace fails while a reader has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def save_manifest(folder: Path, manifest: Dict[str, Any]) -> None:
    _atomic_write(manifest_path(folder), json.dumps(manifest, indent=2, ensure_ascii=False))


def new_suite_id(reports_dir: Path) -> str:
    when = datetime.now()
    while (reports_dir / f"suite_{when:%Y%m%d_%H%M%S}").exists():
        when = datetime.fromtimestamp(when.timestamp() + 1)
    return f"suite_{when:%Y%m%d_%H%M%S}"


def _target_info(target_cfg, key: str) -> Dict[str, Any]:
    info = target_cfg.get_target(key)
    dev = target_cfg.device_profile(key)
    return {
        "key": key,
        "device": dev.device,
        "description": info.get("description", ""),
        "arch": info.get("arch"),
        "accelerator": info.get("accelerator"),
        "simulated": bool(dev.simulate),
        "compute_unit": dev.compute_unit,
        "runtime": dev.runtime,
        "cpu_cores": dev.cpu_cores,
        "cpu_single_core_score": dev.cpu_single_core_score,
        "cpu_scale": round(dev.cpu_scale, 3),
        "min_inference_ms": dev.min_inference_ms,
        "ram_limit_mb": info.get("ram_limit_mb"),
        "vram_limit_mb": info.get("vram_limit_mb"),
        "calibration": {"model": dev.ref_model, "gflops": dev.ref_gflops, "latency_ms": dev.ref_latency_ms,
                        "source": dev.ref_source} if dev.ref_gflops else None,
    }


def cell_note(manifest: Dict[str, Any], model: str, target: str) -> str:
    """User note of a model/device pair: the cell's own "notes" (the one to edit in suite.json), else the original
    input in config.notes; "" for suites created before notes existed."""
    cell = (manifest["cells"].get(model) or {}).get(target) or {}
    if "notes" in cell:
        return clean_note(cell["notes"])
    return note_for(manifest["config"].get("notes") or {}, model, target)


def new_manifest(suite_id: str, config: Dict[str, Any], skipped: List[Dict[str, str]], target_cfg,
                 models_dir: Path) -> Dict[str, Any]:
    """config["notes"] ({model file: {target: note}}, optional) is copied into every cell as "notes"."""
    targets, models = config["targets"], config["models"]
    notes = config.get("notes") or {}
    skipped_names = {s["model"]: s["reason"] for s in skipped}
    all_models = models + [s["model"] for s in skipped if s["model"] not in models]
    config = {**config, "models": all_models}
    cells: Dict[str, Dict[str, Any]] = {}
    for m in all_models:
        cells[m] = {}
        for t in targets:
            if m in skipped_names:
                cells[m][t] = {"status": "skipped", "error": skipped_names[m], "notes": note_for(notes, m, t)}
            else:
                cells[m][t] = {"status": "pending", "notes": note_for(notes, m, t)}
    models_info = {}
    for m in all_models:
        p = models_dir / m
        models_info[m] = {"file": m, "format": Path(m).suffix.lower(),
                          "size_mb": round(p.stat().st_size / 1024 ** 2, 2) if p.exists() else None}
    return {
        "suite_id": suite_id,
        "created": now_iso(),
        "status": "running",
        "config": config,
        "host": target_cfg.simulation.get("host") or {},
        "environment": None,
        "targets_info": {t: _target_info(target_cfg, t) for t in targets},
        "models_info": models_info,
        "cells": cells,
        "timings": {"started": now_iso(), "finished": None, "elapsed_s": 0.0},
    }


def cell_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Key metrics of one run, copied from its report.json into the manifest."""
    est, host, rt = report["device_estimate"], report["host_performance"], report["realtime"]
    res, dets, cold = report["resources"], report["detections"], report["cold_start"]
    sim = est is not None
    stages = host.get("stages") or {}
    stage_mean = lambda k: ((stages.get(k) or {}).get("mean") or 0.0)
    fps = est["est_fps"] if sim else host["throughput_fps"]
    lat = est["latency_ms"] if sim else host["latency_ms"]
    limit, peak = res.get("ram_limit_mb"), res.get("peak_ram_mb")
    return {
        "simulated": sim,
        "fps": fps,
        "est_fps": est["est_fps"] if sim else None,
        "est_latency_p50_ms": est["latency_ms"]["p50"] if sim else None,
        "est_latency_p95_ms": est["latency_ms"]["p95"] if sim else None,
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "host_fps": host["throughput_fps"],
        "host_latency_p50_ms": host["latency_ms"]["p50"],
        "host_latency_p95_ms": host["latency_ms"]["p95"],
        "cpu_side_ms": est["cpu_side_ms"] if sim else round(stage_mean("preprocess") + stage_mean("postprocess"), 2),
        "inference_ms": est["inference_ms"] if sim else round(stage_mean("inference"), 2),
        "peak_ram_mb": peak,
        "ram_limit_mb": limit,
        "ram_pct": round(100.0 * peak / limit, 1) if limit and peak is not None else None,
        "dets_per_frame": dets["mean_per_frame"],
        "frames_with_detections_pct": dets["frames_with_detections_pct"],
        "cold_start_ms": cold.get("model_load_ms"),
        "first_inference_ms": cold.get("first_inference_ms"),
        "output_format": report["model"]["output_format"],
        "required_fps": rt["required_fps"],
        "realtime_capable": rt["realtime_capable"],
        "realtime_factor": rt["device_realtime_factor"],
        "verdict": report["verdict"]["overall"],
        "verdict_items": report["verdict"]["items"],
        "sweep": sweep_mod.best_summary(report),
    }


def heat(cell: Dict[str, Any], required_default: float = 25.0) -> Optional[str]:
    """green = reaches the required FPS, amber >= AMBER_FPS, red below; None when the cell has no result."""
    if cell.get("status") != "ok" or cell.get("fps") is None:
        return None
    required = cell.get("required_fps") or required_default
    fps = cell["fps"]
    return "green" if fps >= required else ("amber" if fps >= AMBER_FPS else "red")


def required_fps_of(manifest: Dict[str, Any]) -> float:
    if manifest["config"].get("required_fps"):
        return float(manifest["config"]["required_fps"])
    for row in manifest["cells"].values():
        for c in row.values():
            if c.get("required_fps"):
                return float(c["required_fps"])
    return 25.0


def iter_cells(manifest: Dict[str, Any]):
    for model in manifest["config"]["models"]:
        for target in manifest["config"]["targets"]:
            yield model, target, manifest["cells"].get(model, {}).get(target) or {"status": "pending"}


def summarize(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Counts per status and the fastest ok pair (simulated devices preferred: x86 is host-measured)."""
    counts = {k: 0 for k in ("ok", "failed", "skipped", "cancelled", "pending", "running")}
    best: Optional[Dict[str, Any]] = None
    for model, target, c in iter_cells(manifest):
        counts[c.get("status", "pending")] = counts.get(c.get("status", "pending"), 0) + 1
        if c.get("status") == "ok" and c.get("fps") is not None:
            key = (bool(c.get("simulated")), c["fps"])
            if best is None or key > best["_key"]:
                best = {"model": model, "target": target, "fps": c["fps"], "simulated": bool(c.get("simulated")), "_key": key}
    if best:
        del best["_key"]
    total = sum(counts.values())
    return {"total": total, "counts": counts, "best": best, "incomplete": counts["pending"] + counts["running"] + counts["cancelled"] > 0}


def finalize_status(manifest: Dict[str, Any]) -> str:
    s = summarize(manifest)
    if s["incomplete"]:
        return "stopped"
    return "completed" if s["counts"]["ok"] > 0 or s["counts"]["skipped"] == s["total"] else "failed"


def mark_stopped(manifest: Dict[str, Any], reason: str = "stopped before this run started") -> None:
    """Pending / running cells of an interrupted suite become 'cancelled'."""
    for model, target, c in iter_cells(manifest):
        if c.get("status") in ("pending", "running"):
            manifest["cells"][model][target] = {"status": "cancelled", "error": reason,
                                                "notes": cell_note(manifest, model, target)}
    seg = manifest["timings"].pop("segment_started_ts", None)  # runner was killed before it could book its time
    if seg:
        manifest["timings"]["elapsed_s"] = round(manifest["timings"].get("elapsed_s", 0.0) + time.time() - seg, 1)
    manifest["status"] = "stopped"
    manifest["timings"]["finished"] = now_iso()


# ---------------------------------------------------------------- exports and rendering

def load_run_report(folder: Path, cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """report.json of a finished cell's run (None when missing or unreadable)."""
    if not cell.get("report_id"):
        return None
    try:
        return json.loads((folder / "runs" / str(cell["report_id"]) / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def flat_rows(manifest: Dict[str, Any], folder: Optional[Path] = None) -> List[Dict[str, Any]]:
    """One row per model/device pair and, for sweep runs (needs the suite folder to read the run reports), per configuration."""
    rows = []
    for model, target, c in iter_cells(manifest):
        base: Dict[str, Any] = {k: c.get(k) for k in CSV_COLUMNS}
        base.update({"model": model, "target": target, "status": c.get("status", "pending"),
                     "notes": cell_note(manifest, model, target),
                     "device": (manifest["targets_info"].get(target) or {}).get("device"),
                     "report_path": f"runs/{c['report_id']}/report.html" if c.get("report_id") else None})
        report = load_run_report(folder, c) if folder is not None and c.get("sweep") else None
        if not report or not report.get("sweep"):
            if c.get("status") == "ok":
                base["is_best"] = True
                best = c.get("sweep") or {}
                if best:
                    base.update({"config_id": best["config_id"], "source_resolution": best["source"], "input_size": best["input"],
                                 "conf": best["conf"], "selection_rule": best["rule"], "selection_reason": best["reason"],
                                 "agreement_f1": best.get("agreement_f1"), "temporal": best.get("temporal"),
                                 "stability": best.get("stability"), "coverage": best.get("coverage"),
                                 "duplicates": best.get("duplicates"), "n_configs": best.get("n_configs")})
            rows.append(base)
            continue
        sw = report["sweep"]
        for e in sorted(sw["configs"], key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"])):
            row = dict(base)
            row.update(cell_metrics(sweep_mod.report_view(report, e)))
            row.update({
                "config_id": e["id"], "source_resolution": e["source"]["label"], "input_size": e["input"]["label"], "conf": e["conf"],
                "is_best": e["id"] == sw["best"]["config_id"], "selection_rule": sw["best"]["rule"] if e["id"] == sw["best"]["config_id"] else "",
                "selection_reason": sw["best"]["reason"] if e["id"] == sw["best"]["config_id"] else "",
                "agreement_f1": (e.get("agreement") or {}).get("f1"), "temporal": e.get("temporal"), "stability": e.get("stability"),
                "coverage": (e.get("ratings") or {}).get("coverage"), "duplicates": (e.get("ratings") or {}).get("duplicates"),
                "n_configs": len(sw["configs"]),
            })
            row["fps"] = row.get("fps")
            rows.append({k: row.get(k) for k in CSV_COLUMNS})
    return rows


def refresh_cells(folder: Path, manifest: Dict[str, Any]) -> bool:
    """Applies the current manual ratings to the sweep run reports of the finished cells (the best configuration may change)
    and copies the new best-configuration metrics into the cells. Returns True when a cell changed."""
    from src.benchmark.export import rerender_bundle
    from src.benchmark.ratings import RatingStore
    changed = False
    video = Path(manifest["config"]["video"]).name if manifest["config"].get("video") else "synthetic"
    rated = {tuple(k.split("|")[:2]) for k in RatingStore().load()}
    for model, target, c in iter_cells(manifest):
        if c.get("status") != "ok" or not c.get("sweep") or not c.get("report_id"):
            continue
        if not c["sweep"].get("rated") and (video, model) not in rated:
            continue  # nothing rated for this video and model, and the report has no rating applied: nothing to refresh
        run_dir = folder / "runs" / str(c["report_id"])
        try:
            report = rerender_bundle(run_dir)
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Could not refresh the ratings of {run_dir.name}: {e}")
            continue
        if report and report.get("sweep"):
            new = cell_metrics(report)
            if new != {k: c.get(k) for k in new}:
                manifest["cells"][model][target] = {**c, **new}
                changed = True
    return changed


def write_exports(folder: Path, manifest: Dict[str, Any]) -> None:
    rows = flat_rows(manifest, folder)
    _atomic_write(folder / "results.json", json.dumps(rows, indent=2, ensure_ascii=False))
    tmp = folder / "results.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel shows non-ASCII notes correctly
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    os.replace(tmp, folder / "results.csv")


def prune_orphan_runs(folder: Path, manifest: Dict[str, Any]) -> None:
    """Run folders not referenced by the manifest (left by a run that was killed while writing) are removed."""
    for stale in (folder / "raw").glob("notes_*.json"):  # temp notes files of a runner that was killed
        stale.unlink(missing_ok=True)
    runs = folder / "runs"
    if not runs.is_dir():
        return
    keep = {c.get("report_id") for _, _, c in iter_cells(manifest) if c.get("report_id")}
    for d in runs.iterdir():
        if d.is_dir() and d.name not in keep:
            shutil.rmtree(d, ignore_errors=True)


def render_suite(folder: Path, manifest: Dict[str, Any]) -> None:
    """Writes the exports and the multipage site. Never raises: a rendering bug must not lose benchmark results."""
    try:
        from src.benchmark.suite_site import write_site
        if refresh_cells(folder, manifest):
            save_manifest(folder, manifest)
        write_exports(folder, manifest)
        write_site(folder, manifest)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not render the suite site: {e}", exc_info=True)


# ---------------------------------------------------------------- runner

def emit(line: str) -> None:
    print(line, flush=True)


class SuiteRunner:
    """Runs the pending cells of a manifest target by target and keeps the manifest and site up to date."""

    def __init__(self, project_root: Path, folder: Path, manifest: Dict[str, Any], local: bool = False):
        self.root = project_root
        self.folder = folder
        self.manifest = manifest
        self.local = local
        self.proc: Optional[subprocess.Popen] = None
        self.container: Optional[str] = None
        self.done = 0
        self.total = 0

    @property
    def rel(self) -> str:
        return f"results/reports/{self.manifest['suite_id']}"

    # -- helpers
    def _cell(self, model: str, target: str) -> Dict[str, Any]:
        return self.manifest["cells"][model][target]

    def _save(self, render: bool = False) -> None:
        save_manifest(self.folder, self.manifest)
        if render:
            render_suite(self.folder, self.manifest)

    def _remove_container(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)

    def kill_current(self) -> None:
        """Stops the running benchmark process and its container (used on Stop / Ctrl+C / SIGTERM)."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
        try:
            self._remove_container()
        except Exception:  # noqa: BLE001
            pass

    def _write_notes_file(self, target: str, models: List[str]) -> Optional[str]:
        """raw/notes_<target>.json with the notes of this target's models (project-relative path), or None when there
        are none. The harness reads it; _run_target deletes it afterwards."""
        notes = {m: {target: n} for m in models if (n := cell_note(self.manifest, m, target))}
        if not notes:
            return None
        path = self.folder / "raw" / f"notes_{target}.json"
        path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
        return f"{self.rel}/raw/{path.name}"

    def _target_command(self, target: str, models: List[str], notes_rel: Optional[str] = None) -> List[str]:
        cfg = self.manifest["config"]
        args = ["--target", target, "--model", *[f"models/{m}" for m in models],
                "--frames", str(cfg["frames"]), "--warmup", str(cfg["warmup"]),
                "--summary-json", f"{self.rel}/raw/{target}.json",
                "--report-root", f"{self.rel}/runs", "--max-samples", "3" if cfg.get("sweep") else "1", "--link-samples",
                "--back-link", "../../index.html", "--no-metrics-files", *sweep_mod.sweep_cli_args(cfg.get("sweep"))]
        if cfg.get("video"):
            args += ["--video", cfg["video"]]
        if cfg.get("conf") is not None:
            args += ["--conf", str(cfg["conf"])]
        if notes_rel:
            args += ["--notes-file", notes_rel]
        if self.local:
            return [sys.executable, str(self.root / "scripts" / "run_benchmark.py"), *args]
        self.container = re.sub(r"[^A-Za-z0-9_.-]", "-", f"crime-detect-{self.manifest['suite_id']}-{target}")
        return build_docker_run_command(target=target, script_name="run_benchmark.py", script_args=args,
                                        project_root=self.root, container_name=self.container)

    # -- cell bookkeeping
    def _finish_cell(self, target: str, model: str, started: float, status: str, report_id: Optional[str] = None,
                     error: Optional[str] = None) -> None:
        c: Dict[str, Any] = {"status": status, "duration_s": round(time.time() - started, 1),
                             "notes": cell_note(self.manifest, model, target)}
        extra = ""
        if status == "ok":
            report = json.loads((self.folder / "runs" / str(report_id) / "report.json").read_text(encoding="utf-8"))
            c.update({"report_id": report_id, **cell_metrics(report)})
            info = self.manifest["models_info"].setdefault(model, {"file": model})
            m = report["model"]
            info.update({k: m[k] for k in ("format", "size_mb", "gflops", "params_m", "input_size", "class_names",
                                           "output_format", "sha256")})
            if not self.manifest.get("environment"):
                self.manifest["environment"] = report["environment"]
            env = report["environment"]
            self.manifest["targets_info"].setdefault(target, {}).setdefault("container", {
                "cpu_limit_cores": env["cgroup"]["cpu_limit_cores"], "memory_limit_mb": env["cgroup"]["memory_limit_mb"],
                "logical_cpus": env["logical_cpus"], "execution": env["execution"]})
            if not self.manifest["config"].get("required_fps"):
                self.manifest["config"]["required_fps"] = report["realtime"]["required_fps"]
            extra = f" fps={c['fps']} heat={heat(c)}"
        else:
            c["error"] = error or "unknown error"
        self.manifest["cells"][model][target] = c
        self.done += 1
        emit(f"SUITE_CELL {self.done}/{self.total} {target} {model} {status}{extra} dur={c['duration_s']}")
        self._save(render=True)

    def _load_summary(self, target: str) -> List[Dict[str, Any]]:
        try:
            return json.loads((self.folder / "raw" / f"{target}.json").read_text(encoding="utf-8")).get("runs", [])
        except (OSError, ValueError):
            return []

    # -- one target
    def _run_target(self, target: str, models: List[str]) -> None:
        (self.folder / "raw").mkdir(exist_ok=True)
        notes_file = self.folder / "raw" / f"notes_{target}.json"
        try:
            self._run_target_inner(target, models)
        finally:
            notes_file.unlink(missing_ok=True)

    def _run_target_inner(self, target: str, models: List[str]) -> None:
        cmd = self._target_command(target, models, self._write_notes_file(target, models))
        emit(f"SUITE_TARGET {target}")
        if self.container:
            emit(f"SUITE_CONTAINER {self.container}")
            self._remove_container()  # a leftover container of an earlier attempt would block the name
        logger.info(f"Target '{target}': {len(models)} model(s) {'(local)' if self.local else 'in container ' + str(self.container)}")
        (self.folder / "raw").mkdir(exist_ok=True)
        try:
            (self.folder / "raw" / f"{target}.json").unlink()
        except OSError:
            pass

        cur: Optional[str] = None
        cur_start = time.time()
        last_error: Dict[str, str] = {}
        tail: Deque[str] = deque(maxlen=6)

        def start_cell(model: str) -> None:
            nonlocal cur, cur_start
            cur, cur_start = model, time.time()
            self._cell(model, target)["status"] = "running"
            emit(f"SUITE_CURRENT {target} {model}")
            self._save()

        def fail_cell(model: str, why: str) -> None:
            self._finish_cell(target, model, cur_start if cur == model else time.time(), "failed",
                              error=last_error.get(model) or why)

        if len(models) == 1:  # a single model prints no "===== [i/n]" marker
            start_cell(models[0])

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1, cwd=str(self.root))
        for raw in self.proc.stdout:
            line = raw.rstrip()
            if PROGRESS_RE.match(line) or SWEEP_PASS_RE.match(line):
                emit(line)
                continue
            print(line, flush=True)
            m = MODEL_LINE_RE.search(line)
            if m:
                if cur is not None:  # previous model ended without a report
                    fail_cell(cur, "the benchmark process ended without a report")
                    cur = None
                if m.group(3) in self.manifest["cells"] and self._cell(m.group(3), target)["status"] != "skipped":
                    start_cell(m.group(3))
                continue
            m = FAIL_LINE_RE.search(line)
            if m:
                last_error[m.group(1)] = m.group(2)[:400]
                continue
            m = REPORT_LINE_RE.search(line)
            if m and cur is not None:
                try:
                    self._finish_cell(target, cur, cur_start, "ok", report_id=m.group(1))
                except (OSError, ValueError, KeyError) as e:
                    fail_cell(cur, f"report could not be read: {e}")
                cur = None
                continue
            if line.strip():
                tail.append(line.strip()[-200:])
        rc = self.proc.wait()
        self.proc = None

        # Cells that never got a REPORT line: use the run summary of the harness, else the process tail
        summary = {Path(str(r.get("model"))).name: r for r in self._load_summary(target)}
        for model in models:
            c = self._cell(model, target)
            if c["status"] not in ("pending", "running"):
                continue
            run = summary.get(model) or {}
            if run.get("report_id") and (self.folder / "runs" / run["report_id"] / "report.json").is_file():
                self._finish_cell(target, model, cur_start, "ok", report_id=run["report_id"])
                continue
            why = run.get("error") or last_error.get(model)
            if not why:
                if rc == 0:
                    why = "the benchmark finished without producing a report"
                else:
                    hint = " (killed: out of memory or the container was removed)" if rc in (137, -9, 143) else ""
                    errs = [t for t in tail if ERROR_HINT_RE.search(t)]
                    why = f"container exited with code {rc}{hint}" + (": " + " | ".join(errs[-2:]) if errs else "")
            fail_cell(model, why)

    # -- whole suite
    def run(self) -> str:
        m = self.manifest
        todo: Dict[str, List[str]] = {}
        for target in m["config"]["targets"]:
            models = [mod for mod in m["config"]["models"] if self._cell(mod, target)["status"] not in DONE_STATUSES]
            for mod in models:
                self.manifest["cells"][mod][target] = {"status": "pending", "notes": cell_note(m, mod, target)}
            if models:
                todo[target] = models
        self.total = sum(len(v) for v in todo.values())
        m["status"] = "running"
        m["timings"]["finished"] = None
        m["timings"]["segment_started_ts"] = time.time()
        self._save(render=True)
        emit(f"SUITE_START {m['suite_id']} {self.total}")
        cfg = m["config"]
        gf = cached_gflops(self.root)
        dyn = cached_dynamic(self.root)
        eta = estimate_seconds([((m["models_info"].get(mod) or {}).get("gflops") or gf.get(mod),
                                 (m["models_info"].get(mod) or {}).get("size_mb") or 0.0,
                                 dyn.get(mod, Path(mod).suffix.lower() == ".pt"), Path(mod).suffix.lower() == ".pt")
                                for mod in sorted({mod for v in todo.values() for mod in v})],
                               len(todo), cfg["frames"], cfg["warmup"], cfg.get("sweep"))
        logger.info(f"{self.total} run(s) on {len(todo)} target(s), {cfg['frames']} frames each. "
                    f"Rough ETA: {eta / 60:.0f} min (varies with model size and CPU budget).")
        began = time.time()
        stopped, crashed = False, False
        try:
            for target, models in todo.items():
                self._run_target(target, models)
        except KeyboardInterrupt:
            stopped = True
            logger.warning("Stop requested: cancelling the remaining runs...")
            self.kill_current()
        except Exception:  # noqa: BLE001  (a runner bug must still leave a consistent, renderable suite)
            crashed = True
            logger.error("Suite runner crashed", exc_info=True)
            self.kill_current()
        finally:
            m["timings"]["elapsed_s"] = round(m["timings"].get("elapsed_s", 0.0) + time.time() - began, 1)
            m["timings"].pop("segment_started_ts", None)
            if stopped or crashed:
                mark_stopped(m, "cancelled: the suite was stopped" if stopped else "cancelled: the suite runner crashed")
                if crashed:
                    m["status"] = "failed"
            else:
                m["timings"]["finished"] = now_iso()
                m["status"] = finalize_status(m)
            prune_orphan_runs(self.folder, m)
            self._save(render=True)
        emit(f"SUITE_DONE {m['suite_id']}")
        if stopped:
            raise KeyboardInterrupt
        return m["status"]
