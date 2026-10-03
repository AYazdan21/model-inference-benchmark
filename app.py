"""
Lightweight Web Application for Model Inference Benchmark.
Runs using Python standard library (zero external dependencies).
Provides:
1. Multi-threaded HTTP server for concurrent requests.
2. Pre-rendered initial data so dropdowns load INSTANTLY (0ms).
3. Cached Docker and Target status to prevent subprocess blocking.
4. Interactive Video Inference player with MJPEG stream.
5. Video playback controls: Play/Pause, Step Forward/Backward, Seek Timeline.
6. Option to save or discard resulting output video.
7. Benchmark profiling mode with live metrics & terminal console.
8. Benchmark sweeps (source resolution x model input size x confidence), best configuration per model/device pair and
   manual quality ratings ("Rate detection quality") that take priority in the best-configuration choice.
"""

import http.server
import json
import io
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TargetConfig, load_yaml
from src.runtimes import get_detector
from src.benchmark import sweep as sweep_mod
from src.benchmark.notes import NOTES_MAX_CHARS, clean_notes_map
from src.benchmark.ratings import COVERAGE_HELP, DUPLICATES_HELP, RatingStore
from src.benchmark.rerender import rerender_matching
from src.video import VideoReader, VideoAnnotator
from src.utils.resource_limiter import setup_resource_limiter
from src.utils.docker_runner import is_docker_daemon_running, build_docker_run_command
from src.benchmark.suite import (SUITE_ID_RE, RUNNABLE_EXTS, CONTAINER_START_S, CELL_OVERHEAD_S, HOST_MS_BASE,
                                 HOST_MS_PER_GFLOP, DEFAULT_GFLOPS, LOAD_S_PER_MB, DECODE_MS_PER_FRAME, PASS_OVERHEAD_S, EXTRA_WARMUP, TORCH_START_S,
                                 heat as suite_heat, load_manifest, summarize)

PORT = 5000

# Benchmark report bundles (written by scripts/run_benchmark.py)
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
NOTES_DIR = REPORTS_DIR / "_notes"  # temp notes files handed to the benchmark processes (deleted when the job ends)
REPORT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REPORT_SAMPLE_RE = re.compile(r"^samples/[A-Za-z0-9_.-]+\.jpg$")
# file -> (content type, sent as attachment)
REPORT_FILES = {
    "report.html": ("text/html; charset=utf-8", False),
    "report.json": ("application/json", False),
    "frames.csv": ("text/csv; charset=utf-8", True),
    "summary.md": ("text/markdown; charset=utf-8", True),
}
# Benchmark suites (scripts/run_benchmark_matrix.py): folders results/reports/suite_<timestamp>/
SUITE_FILES = {  # extension -> (content type, sent as attachment)
    ".html": ("text/html; charset=utf-8", False),
    ".json": ("application/json", False),
    ".csv": ("text/csv; charset=utf-8", True),
    ".md": ("text/markdown; charset=utf-8", True),
    ".jpg": ("image/jpeg", False),
}
SUITE_CSP = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
SUITE_LOG_LINES = 400  # a suite prints thousands of lines: keep the tail only
SUITE_START_RE = re.compile(r"^SUITE_START (\S+) (\d+)\s*$")
SUITE_CELL_RE = re.compile(r"^SUITE_CELL (\d+)/(\d+) (\S+) (.+?) (ok|failed|skipped|cancelled)((?: \w+=\S+)*)\s*$")
SUITE_CURRENT_RE = re.compile(r"^SUITE_CURRENT (\S+) (.+?)\s*$")
SWEEP_PASS_RE = re.compile(r"^SWEEP_PASS (\d+)/(\d+) (\S+) (\S+)\s*$")
RATING_SOURCE_RE = re.compile(r"^(native|\d{3,4}p)$")
RATING_INPUT_RE = re.compile(r"^\d{2,4}x\d{2,4}$")
ratings_lock = threading.Lock()
PROGRESS_RE = re.compile(r"^PROGRESS (\d+)/(\d+)\s*$")
REPORT_LINE_RE = re.compile(r"\] REPORT ([A-Za-z0-9_.-]+)\s*$")
MODEL_LINE_RE = re.compile(r"=====\s*\[(\d+)/(\d+)\]\s*(.+?)\s*=====")

# Cache Docker status to prevent blocking requests with repetitive 'docker info' calls
cached_docker_status = False

def check_docker_status_cached():
    global cached_docker_status
    cached_docker_status = is_docker_daemon_running()
    return cached_docker_status

# Threaded HTTP Server to handle concurrent streaming and API requests
class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

# Global Live Inference Session State
class LiveInferenceSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.paused = False
        self.stop_requested = False
        self.process: Optional[subprocess.Popen] = None
        self.reader: Optional[VideoReader] = None
        self.annotator: Optional[VideoAnnotator] = None
        self.detector = None
        self.writer = None
        self.latest_frame_jpeg: Optional[bytes] = None
        self.viewers = 0  # open /api/video_feed connections (managed by the feed handler, not reset)
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.latency_ms = 0.0
        self.host_latency_ms = 0.0
        self.simulated = False
        self.detections_count = 0
        self.save_output = True
        self.output_path = ""
        self.seek_target_frame: Optional[int] = None

    def reset(self):
        with self.lock:
            self.active = False
            self.paused = False
            self.stop_requested = False
            if self.process:
                try:
                    self.process.terminate()
                except Exception:
                    pass
                self.process = None
            if self.reader:
                try:
                    self.reader.release()
                except Exception:
                    pass
                self.reader = None
            if self.writer:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
            self.detector = None
            self.latest_frame_jpeg = None
            self.current_frame = 0
            self.total_frames = 0
            self.seek_target_frame = None

live_session = LiveInferenceSession()

job_state = {
    "running": False,
    "process": None,
    "logs": [],
    "metrics": None,
    "last_run_type": None,
    "container": None,     # name of the docker container running the current benchmark (removed on Stop)
    "progress": None,      # {"done", "total", "model_index", "model_count", "model"} while a benchmark runs
    "report_ids": [],      # report bundles produced by the current/last benchmark
    "suite": None,         # live state of the current/last benchmark suite (see new_suite_state)
}
state_lock = threading.Lock()

def get_available_models():
    models_dir = PROJECT_ROOT / "models"
    valid_exts = {".pt", ".onnx", ".engine", ".rknn", ".hef", ".trt"}
    models = []
    if models_dir.exists():
        for f in sorted(models_dir.glob("*")):
            if f.suffix.lower() in valid_exts:
                models.append(f.name)
    # Ensure preferred model is first
    pref = "HumanDetection_light_input_640.onnx"
    if pref in models:
        models.remove(pref)
        models.insert(0, pref)
    return models or ["HumanDetection_light_input_640.onnx", "best-yolo11-seg.pt"]

_flag_cache: dict = {}

def model_flag(path: Path) -> Optional[str]:
    """Reason a model cannot run, or None. Detects ONNX graphs whose external weight file is missing
    (small graph-only files reference it as 'location'), without needing the onnx package."""
    ext = path.suffix.lower()
    if ext not in RUNNABLE_EXTS:
        return "device-native format: needs the vendor runtime on real hardware (skipped)"
    if ext != ".onnx":
        return None
    try:
        st = path.stat()
        key = (str(path), st.st_mtime)
        if key in _flag_cache:
            return _flag_cache[key]
        reason = None
        if st.st_size < 8 * 1024 * 1024:
            data = path.read_bytes()
            for m in re.finditer(rb"\x0a\x08location\x12([\x01-\x7f])", data):
                name = data[m.end():m.end() + m.group(1)[0]].decode("utf-8", errors="replace")
                if name and not (path.parent / name).exists():
                    reason = f"broken model: external weight file '{name}' is missing"
                    break
        _flag_cache[key] = reason
        return reason
    except OSError:
        return None

def get_model_flags() -> dict:
    models_dir = PROJECT_ROOT / "models"
    flags = {}
    for name in get_available_models():
        reason = model_flag(models_dir / name)
        if reason:
            flags[name] = reason
    return flags

def get_model_sizes() -> dict:
    models_dir = PROJECT_ROOT / "models"
    sizes = {}
    for name in get_available_models():
        try:
            sizes[name] = round((models_dir / name).stat().st_size / 1024 ** 2, 1)
        except OSError:
            pass
    return sizes

def get_available_videos():
    video_dir = PROJECT_ROOT / "videos" / "input"
    valid_exts = {".mp4", ".avi", ".mkv", ".mov"}
    videos = []
    if video_dir.exists():
        for f in sorted(video_dir.glob("*")):
            if f.suffix.lower() in valid_exts:
                videos.append(f.name)
    return videos

def get_current_status():
    try:
        target_cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")
        targets = target_cfg.targets
    except Exception:
        targets = {}

    return {
        "targets": targets,
        "models": get_available_models(),
        "videos": get_available_videos(),
        "model_flags": get_model_flags(),
        "suite_estimate": {"container_start_s": CONTAINER_START_S, "cell_overhead_s": CELL_OVERHEAD_S,
                           "host_ms_base": HOST_MS_BASE, "host_ms_per_gflop": HOST_MS_PER_GFLOP,
                           "default_gflops": DEFAULT_GFLOPS, "load_s_per_mb": LOAD_S_PER_MB, "decode_ms": DECODE_MS_PER_FRAME,
                           "pass_overhead_s": PASS_OVERHEAD_S, "extra_warmup": EXTRA_WARMUP, "torch_start_s": TORCH_START_S},
        "model_sizes": get_model_sizes(),
        "sweep": sweep_defaults(),
        "docker_available": cached_docker_status,
        "running": bool(job_state["running"] or live_session.active),
        "last_run_type": job_state["last_run_type"],
        "has_metrics": job_state["metrics"] is not None
    }

def parse_notes(raw, models: list, targets: list):
    """Notes of a benchmark request as a cleaned {model: {target: text}} limited to the requested models and targets,
    or an error message when the structure is wrong. Texts are stripped and cut at NOTES_MAX_CHARS characters."""
    try:
        cleaned = clean_notes_map(raw)
    except ValueError as e:
        return str(e)
    out = {}
    for model, per_target in cleaned.items():
        picked = {t: n for t, n in per_target.items() if t in targets}
        if model in models and picked:
            out[model] = picked
    return out

def detection_config() -> dict:
    try:
        return load_yaml(PROJECT_ROOT / "configs" / "detection.yaml")
    except Exception:
        return {}

def sweep_defaults() -> dict:
    """Defaults, presets and the confidence of configs/detection.yaml for the sweep panels of the page."""
    cfg = detection_config()
    conf = float((cfg.get("model") or {}).get("conf_threshold", 0.35))
    base = (cfg.get("benchmark") or {}).get("sweep") or {}
    try:
        defaults = sweep_mod.normalize_sweep({}, conf, base)
    except ValueError:
        defaults = sweep_mod.normalize_sweep({}, conf)
    return {"defaults": defaults, "presets": {k: sweep_mod.normalize_sweep(v, conf, base) for k, v in sweep_mod.PRESETS.items()},
            "default_conf": conf, "height_choices": list(sweep_mod.SOURCE_HEIGHT_CHOICES), "size_choices": list(sweep_mod.INPUT_SIZE_CHOICES)}

def parse_sweep(raw, conf):
    """Validated sweep settings of a request ({enabled: False} when switched off) or an error message.
    A request without sweep settings gets the defaults of configs/detection.yaml."""
    cfg = detection_config()
    default_conf = conf if conf is not None else float((cfg.get("model") or {}).get("conf_threshold", 0.35))
    if raw is not None and not isinstance(raw, dict):
        return "sweep must be an object"
    try:
        return sweep_mod.normalize_sweep(raw, default_conf, (cfg.get("benchmark") or {}).get("sweep"))
    except ValueError as e:
        return f"Invalid sweep settings: {e}"

def write_notes_file(notes: dict) -> Path:
    """results/reports/_notes/notes_<timestamp>.json (UTF-8): text never goes through command-line arguments."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTES_DIR / f"notes_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
    return path

def remove_notes_file(path: Optional[Path]):
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
        NOTES_DIR.rmdir()  # only succeeds when no other job still has a notes file in there
    except OSError:
        pass

def resolve_report_folder(report_id: str) -> Optional[Path]:
    """Folder of a report id, or None if the id is malformed, missing or would escape results/reports."""
    if not REPORT_ID_RE.match(report_id) or report_id.startswith("."):
        return None
    root = REPORTS_DIR.resolve()
    folder = (root / report_id).resolve()
    if folder.parent != root or not (folder / "report.json").is_file():
        return None
    return folder

def list_reports() -> list:
    """Summaries of all report bundles in results/reports, newest first (unreadable ones are skipped)."""
    rows = []
    if not REPORTS_DIR.is_dir():
        return rows
    for folder in REPORTS_DIR.iterdir():
        report_json = folder / "report.json"
        if not report_json.is_file():
            continue
        try:
            r = json.loads(report_json.read_text(encoding="utf-8"))
            est = r.get("device_estimate") or {}
            created = r["created"]
            try:
                stamp = datetime.fromisoformat(created).timestamp()
            except ValueError:
                stamp = report_json.stat().st_mtime
            rows.append({
                "report_id": r["report_id"],
                "created": created,
                "target": r["target"]["key"],
                "device": r["target"]["device"],
                "model": r["model"]["file"],
                "notes": r.get("notes") or "",
                "simulated": r["target"]["simulated"],
                "est_fps": est.get("est_fps"),
                "host_fps": r["host_performance"]["throughput_fps"],
                "detections_per_frame": r["detections"]["mean_per_frame"],
                "peak_ram_mb": r["resources"]["peak_ram_mb"],
                "best_config": sweep_mod.best_summary(r) if r.get("sweep") else None,
                "frames": r["config"]["frames_processed"],
                "overall": r["verdict"]["overall"],
                "_stamp": stamp,
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    rows.sort(key=lambda row: row["_stamp"], reverse=True)
    for row in rows:
        del row["_stamp"]
    return rows

def _device_cell(dev: dict, entry: dict) -> Optional[dict]:
    """FPS, real-time flag and best-configuration flag of one configuration on one device (None if that run lacks it)."""
    sw = dev["report"]["sweep"]
    x = next((c for c in sw["configs"] if c["id"] == entry["id"]), None)
    if x is None:
        return None
    return {"fps": round(sweep_mod.entry_fps(x), 1), "realtime": bool(x["realtime"]["realtime_capable"]),
            "best": sw["best"]["config_id"] == entry["id"]}

def _view_item(report: dict, base_url: str, devices: list, ratings: dict) -> dict:
    """Rating-view entry of one model: its configurations (same sample frames in all), metrics and the current ratings."""
    sw = report["sweep"]
    model, video = report["model"]["file"], sw.get("video") or "synthetic"
    configs = []
    for e in sorted(sw["configs"], key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"])):
        r = ratings.get(f"{video}|{model}|{e['source']['label']}|{e['input']['label']}|{sweep_mod.conf_text(e['conf'])}") or {}
        configs.append({
            "id": e["id"], "label": e["label"], "short": e["short"], "source": e["source"]["label"], "source_text": e["source"]["text"],
            "input": e["input"]["label"], "conf": e["conf"], "samples": e.get("samples") or [],
            "stability": e.get("stability"), "agreement_f1": (e.get("agreement") or {}).get("f1"), "temporal": e.get("temporal"),
            "dets_per_frame": e["detections"]["mean_per_frame"], "coverage": r.get("coverage"), "duplicates": r.get("duplicates"),
            "per_device": {d["target"]: _device_cell(d, e) for d in devices},
        })
    return {"model": model, "video": video, "base_url": base_url, "sample_frames": sw.get("sample_frames") or [],
            "locked": bool(sw["dimensions"].get("input_locked")),
            "devices": [{"target": d["target"], "device": d["device"], "best": d["report"]["sweep"]["best"]["label"],
                         "rule": d["report"]["sweep"]["best"]["rule"]} for d in devices],
            "configs": configs}

def build_ratings_view(report_id: str, suite_id: str) -> dict:
    """Data of the rating view of a report or a suite (one item per model)."""
    ratings = RatingStore().load()
    help_txt = {"coverage": COVERAGE_HELP, "duplicates": DUPLICATES_HELP}
    if report_id:
        folder = resolve_report_folder(report_id)
        if folder is None:
            raise FileNotFoundError("Report not found")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if not report.get("sweep"):
            raise ValueError("This report was made before benchmark sweeps: it has a single configuration, so there is nothing to compare or rate.")
        dev = [{"target": report["target"]["key"], "device": report["target"]["device"], "report": report}]
        return {"kind": "report", "id": report_id, "help": help_txt,
                "items": [_view_item(report, f"/api/reports/{report_id}/", dev, ratings)]}
    folder = resolve_suite_folder(suite_id)
    if folder is None:
        raise FileNotFoundError("Suite not found")
    manifest = load_manifest(folder)
    if not (manifest["config"].get("sweep")):
        raise ValueError("This suite was run without a sweep: it has a single configuration per pair, so there is nothing to compare or rate.")
    items = []
    for model in manifest["config"]["models"]:
        devs = []
        for target in manifest["config"]["targets"]:
            c = (manifest["cells"].get(model) or {}).get(target) or {}
            if c.get("status") == "ok" and c.get("sweep") and c.get("report_id"):
                try:
                    r = json.loads((folder / "runs" / c["report_id"] / "report.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if r.get("sweep"):
                    devs.append({"target": target, "device": (manifest["targets_info"].get(target) or {}).get("device", target),
                                 "report": r, "report_id": c["report_id"]})
        if devs:
            items.append(_view_item(devs[0]["report"], f"/api/suites/{suite_id}/runs/{devs[0]['report_id']}/", devs, ratings))
    if not items:
        raise ValueError("No finished sweep run in this suite yet.")
    return {"kind": "suite", "id": suite_id, "help": help_txt, "items": items}

def resolve_suite_folder(suite_id: str) -> Optional[Path]:
    """Folder of a suite id, or None if the id is malformed, missing or would escape results/reports."""
    if not SUITE_ID_RE.match(suite_id):
        return None
    root = REPORTS_DIR.resolve()
    folder = (root / suite_id).resolve()
    if folder.parent != root or not (folder / "suite.json").is_file():
        return None
    return folder

def active_suite_id() -> Optional[str]:
    with state_lock:
        st = job_state.get("suite")
        if job_state["running"] and job_state["last_run_type"] == "suite" and st:
            return st.get("suite_id")
    return None

def suite_summary(folder: Path, active_id: Optional[str]) -> Optional[dict]:
    try:
        m = load_manifest(folder)
        s = summarize(m)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    status = m["status"]
    if status == "running" and m["suite_id"] != active_id:
        status = "stopped"  # the runner died without finishing (app or machine restarted)
    cfg = m["config"]
    return {
        "suite_id": m["suite_id"], "created": m["created"], "status": status,
        "n_models": len(cfg["models"]), "n_devices": len(cfg["targets"]), "frames": cfg["frames"],
        "video": Path(cfg["video"]).name if cfg.get("video") else None,
        "counts": s["counts"], "total": s["total"], "best": s["best"], "elapsed_s": m["timings"].get("elapsed_s"),
        "resumable": s["incomplete"] or status in ("stopped", "failed"),
        "has_report": (folder / "index.html").is_file(),
        "has_sweep": bool(cfg.get("sweep")),
    }

def list_suites() -> list:
    """Summaries of all suites in results/reports, newest first."""
    rows = []
    if not REPORTS_DIR.is_dir():
        return rows
    active_id = active_suite_id()
    for folder in REPORTS_DIR.iterdir():
        if SUITE_ID_RE.match(folder.name) and (folder / "suite.json").is_file():
            row = suite_summary(folder, active_id)
            if row:
                rows.append(row)
    rows.sort(key=lambda r: r["created"], reverse=True)
    return rows

def new_suite_state(models: list, targets: list, suite_id: Optional[str] = None) -> dict:
    return {
        "suite_id": suite_id, "status": "starting", "total": 0, "done": 0, "models": models, "targets": targets,
        "cells": {m: {t: {"status": "pending"} for t in targets} for m in models},
        "current": None, "target": None, "cell_seconds": [], "stopping": False,
        "started_ts": time.time(), "ended_ts": None,
    }

def parse_suite_request(req: dict):
    """Validated suite settings, or an error message."""
    try:
        cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")
        all_targets = cfg.available_targets
    except Exception as e:
        return f"Cannot read targets.yaml: {e}"
    models_avail = get_available_models()
    targets = req.get("targets") or all_targets
    models = req.get("models") or [m for m in models_avail if Path(m).suffix.lower() in RUNNABLE_EXTS]
    if not isinstance(targets, list) or not isinstance(models, list):
        return "targets and models must be lists"
    bad_t = [t for t in targets if t not in all_targets]
    bad_m = [m for m in models if m not in models_avail]
    if bad_t or bad_m or not targets or not models:
        return f"Unknown or empty selection (targets {bad_t}, models {bad_m})"
    video = req.get("video") or ""
    if video and video not in get_available_videos():
        return f"Unknown video '{video}'"
    env = req.get("env", "docker")
    if env not in ("docker", "local"):
        return "env must be 'docker' or 'local'"
    try:
        frames = max(1, min(2000, int(req.get("frames", 30))))
        warmup = max(0, min(50, int(req.get("warmup", 3))))
        conf = req.get("conf")
        conf = None if conf in (None, "") else max(0.01, min(0.99, float(conf)))
    except (TypeError, ValueError):
        return "frames, warmup and conf must be numbers"
    notes = parse_notes(req.get("notes"), models, targets)
    if isinstance(notes, str):
        return notes
    sweep = parse_sweep(req.get("sweep"), conf)
    if isinstance(sweep, str):
        return sweep
    return {"env": env, "targets": targets, "models": models, "video": video, "frames": frames, "warmup": warmup, "conf": conf,
            "notes": notes, "sweep": sweep}

class AppHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            status_json = json.dumps(get_current_status())
            page = HTML_PAGE.replace('"__INITIAL_DATA_PLACEHOLDER__"', status_json)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
        elif path == "/api/status":
            self.send_json(get_current_status())
        elif path == "/api/logs":
            self.handle_api_logs()
        elif path == "/api/video_feed":
            self.handle_video_feed()
        elif path == "/api/video/status":
            self.handle_video_status()
        elif path == "/api/reports":
            self.send_json({"reports": list_reports()})
        elif path == "/api/estimates":
            self.handle_cached_estimates()
        elif path == "/api/suites":
            self.send_json({"suites": list_suites()})
        elif path == "/api/ratings/view":
            self.handle_ratings_view(parse_qs(parsed.query))
        elif path.startswith("/api/suites/"):
            self.handle_suite_file(path)
        elif path.startswith("/api/reports/"):
            self.handle_report_file(path)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            self.handle_api_run()
        elif path == "/api/benchmark":
            self.handle_api_run(force_mode="benchmark")
        elif path == "/api/estimates":
            self.handle_api_estimates()
        elif path == "/api/ratings":
            self.handle_api_ratings()
        elif path == "/api/suite":
            self.handle_api_suite()
        elif path == "/api/suite/resume":
            self.handle_api_suite(resume=True)
        elif path == "/api/stop":
            self.handle_api_stop()
        elif path == "/api/video/playback":
            self.handle_video_playback()
        elif path == "/api/video/seek":
            self.handle_video_seek()
        else:
            self.send_error(404, "Not Found")

    def handle_api_logs(self):
        with state_lock:
            suite = json.loads(json.dumps(job_state["suite"])) if job_state["suite"] else None
            if suite:
                suite["now"] = time.time()
            data = {
                "running": bool(job_state["running"] or live_session.active),
                "logs": list(job_state["logs"]),
                "metrics": job_state["metrics"],
                "progress": job_state["progress"],
                "report_ids": list(job_state["report_ids"]),
                "last_run_type": job_state["last_run_type"],
                "suite": suite,
            }
        self.send_json(data)

    def handle_video_status(self):
        with live_session.lock:
            cur = live_session.current_frame
            total = live_session.total_frames
            fps = live_session.fps or 30.0
            cur_sec = cur / fps if fps > 0 else 0
            total_sec = total / fps if fps > 0 else 0

            cur_str = time.strftime("%M:%S", time.gmtime(cur_sec))
            total_str = time.strftime("%M:%S", time.gmtime(total_sec))
            pct = (cur / total * 100.0) if total > 0 else 0.0

            data = {
                "active": live_session.active,
                "paused": live_session.paused,
                "current_frame": cur,
                "total_frames": total,
                "fps": round(fps, 1),
                "latency_ms": round(live_session.latency_ms, 1),
                "host_latency_ms": round(live_session.host_latency_ms, 1),
                "simulated": live_session.simulated,
                "detections": live_session.detections_count,
                "current_time": cur_str,
                "total_time": total_str,
                "progress_pct": round(pct, 2),
                "saved": live_session.save_output,
                "output_path": live_session.output_path,
                "viewers": live_session.viewers
            }
        self.send_json(data)

    def handle_video_playback(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)
        action = req.get("action", "toggle")

        with live_session.lock:
            if action == "pause":
                live_session.paused = True
            elif action == "play":
                live_session.paused = False
            elif action == "toggle":
                live_session.paused = not live_session.paused

            if live_session.process and live_session.process.stdin:
                try:
                    cmd_str = b"pause\n" if live_session.paused else b"resume\n"
                    live_session.process.stdin.write(cmd_str)
                    live_session.process.stdin.flush()
                except Exception:
                    pass

        self.send_json({"paused": live_session.paused})

    def handle_video_seek(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)

        target_frame = None
        with live_session.lock:
            fps = live_session.fps or 30.0
            total = live_session.total_frames
            cur = live_session.current_frame

            if "frame" in req:
                target_frame = int(req["frame"])
            elif "offset_seconds" in req:
                offset_frames = int(float(req["offset_seconds"]) * fps)
                target_frame = cur + offset_frames
            elif "pct" in req:
                target_frame = int((float(req["pct"]) / 100.0) * total)

            if target_frame is not None:
                clamped = max(0, min(target_frame, max(0, total - 1)))
                live_session.seek_target_frame = clamped
                if live_session.process and live_session.process.stdin:
                    try:
                        live_session.process.stdin.write(f"seek:{clamped}\n".encode("utf-8"))
                        live_session.process.stdin.flush()
                    except Exception:
                        pass

        self.send_json({"status": "seek_queued", "target_frame": target_frame})

    def handle_video_feed(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        blank_jpeg = None
        # The browser may connect before the worker has started the session (container start + model load),
        # so wait for it instead of closing the stream immediately, which left the player black.
        deadline = time.time() + 120
        started = False
        with live_session.lock:
            live_session.viewers += 1
        try:
            while True:
                with live_session.lock:
                    active = live_session.active
                    frame_data = live_session.latest_frame_jpeg
                if active:
                    started = True
                else:
                    with state_lock:
                        pending = job_state["running"] and job_state["last_run_type"] == "inference"
                    if started or not pending or time.time() > deadline:
                        break

                if frame_data:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                else:
                    if blank_jpeg is None:
                        blank = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "Initializing frame...", (190, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 160), 2)
                        _, buf = cv2.imencode(".jpg", blank)
                        blank_jpeg = buf.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(blank_jpeg)
                    self.wfile.write(b"\r\n")

                time.sleep(0.04)
        except OSError:
            pass  # viewer disconnected (reset / aborted / broken pipe)
        finally:
            with live_session.lock:
                live_session.viewers -= 1

    def handle_api_run(self, force_mode: Optional[str] = None):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body)

        mode = force_mode or req.get("mode", "inference")
        if mode != "inference":  # benchmark notes: validated here so a bad request is refused before a job starts
            notes = parse_notes(req.get("notes"), req.get("models") or [req.get("model", "")], [req.get("target", "x86-cpu")])
            if isinstance(notes, str):
                self.send_json({"error": notes}, status=400)
                return
            req["notes"] = notes
            sweep = parse_sweep(req.get("sweep"), float(req["conf"]) if req.get("conf") not in (None, "") else None)
            if isinstance(sweep, str):
                self.send_json({"error": sweep}, status=400)
                return
            req["sweep"] = sweep
        with state_lock:
            if job_state["running"] or live_session.active:
                self.send_json({"error": "A session is already running"}, status=400)
                return
            job_state["running"] = True
            job_state["logs"] = []
            job_state["metrics"] = None
            job_state["last_run_type"] = mode
            job_state["progress"] = None
            job_state["report_ids"] = []

        if mode == "inference":
            thread = threading.Thread(target=run_live_inference_worker, args=(req,), daemon=True)
            thread.start()
        else:
            thread = threading.Thread(target=run_job_worker, args=(req,), daemon=True)
            thread.start()

        self.send_json({"status": "started"})

    def handle_api_stop(self):
        with live_session.lock:
            live_session.stop_requested = True
            if live_session.process:
                try:
                    if live_session.process.stdin:
                        live_session.process.stdin.write(b"stop\n")
                        live_session.process.stdin.flush()
                except Exception:
                    pass
                try:
                    live_session.process.terminate()
                except Exception:
                    pass
        with state_lock:
            if job_state["running"] and job_state["last_run_type"] == "suite" and job_state["suite"]:
                # The suite worker renders the partial report after the runner is gone, then clears "running"
                job_state["suite"]["stopping"] = True
                job_state["logs"].append("[APP] Stop requested: cancelling the suite...")
                if job_state["process"]:
                    try:
                        job_state["process"].terminate()
                    except Exception:
                        pass
                if job_state["container"]:
                    try:
                        subprocess.Popen(["docker", "rm", "-f", job_state["container"]],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass
                self.send_json({"status": "stopping"})
                return
            if job_state["process"]:
                try:
                    job_state["process"].terminate()
                except Exception:
                    pass
                if job_state["container"]:
                    # terminate() only stops the compose client; remove the container so the benchmark really stops
                    try:
                        subprocess.Popen(["docker", "rm", "-f", job_state["container"]],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass
                    job_state["container"] = None
                job_state["process"] = None
                job_state["logs"].append("[APP] Process termination requested by user.")
                job_state["running"] = False
        self.send_json({"status": "stopped"})

    def handle_api_estimates(self):
        """Static model x target latency table (no inference). Runs where the ONNX tooling is installed."""
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        out_rel = "results/metrics/_estimates.json"
        if req.get("env", "docker") == "docker":
            cmd = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker" / "docker-compose.yml"),
                   "run", "--rm", "-T", "x86-cpu", "python", "scripts/estimate_models.py", "--json", out_rel]
        else:
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "estimate_models.py"), "--json", str(PROJECT_ROOT / out_rel)]
        try:
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
            out_path = PROJECT_ROOT / out_rel
            if res.returncode != 0 or not out_path.exists():
                tail = (res.stderr or res.stdout).strip().splitlines()[-5:]
                self.send_json({"error": "Estimation failed: " + " | ".join(tail)}, status=500)
                return
            self.send_json(json.loads(out_path.read_text(encoding="utf-8")))
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_cached_estimates(self):
        """GET /api/estimates: the table of the last 'Estimate' run (used for the suite ETA), {} if there is none."""
        out_path = PROJECT_ROOT / "results" / "metrics" / "_estimates.json"
        try:
            self.send_json(json.loads(out_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.send_json({})

    def handle_api_suite(self, resume: bool = False):
        """POST /api/suite (new suite) and /api/suite/resume {suite_id}: starts the suite runner in a worker."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self.send_json({"error": "Invalid JSON"}, status=400)
            return
        if resume:
            folder = resolve_suite_folder(str(req.get("suite_id", "")))
            if folder is None:
                self.send_json({"error": "Suite not found"}, status=404)
                return
            try:
                manifest = load_manifest(folder)
            except (OSError, ValueError):
                self.send_json({"error": "Suite manifest unreadable"}, status=500)
                return
            cfg = manifest["config"]
            settings = {"env": cfg.get("execution", "docker"), "resume": manifest["suite_id"]}
            models, targets = cfg["models"], cfg["targets"]
        else:
            settings = parse_suite_request(req)
            if isinstance(settings, str):
                self.send_json({"error": settings}, status=400)
                return
            models, targets = settings["models"], settings["targets"]
        if settings["env"] == "docker" and not is_docker_daemon_running():
            self.send_json({"error": "Docker daemon is not running. Start Docker Desktop or switch to Host Python."}, status=400)
            return
        state = new_suite_state(models, targets, settings.get("resume"))
        with state_lock:
            if job_state["running"] or live_session.active:
                self.send_json({"error": "A session is already running (stop it first: a suite needs the machine to itself)"}, status=400)
                return
            job_state["running"] = True
            job_state["logs"] = []
            job_state["metrics"] = None
            job_state["last_run_type"] = "suite"
            job_state["progress"] = None
            job_state["report_ids"] = []
            job_state["suite"] = state
        threading.Thread(target=run_suite_worker, args=(settings, state), daemon=True).start()
        self.send_json({"status": "started"})

    def handle_suite_file(self, path: str):
        """GET /api/suites/<id>/<relative file>, /api/suites/<id>/download (ZIP of the whole suite)."""
        # Split before decoding so an encoded slash cannot smuggle extra path segments
        segments = [unquote(seg) for seg in path[len("/api/suites/"):].split("/")]
        suite_id, rel_parts = segments[0], segments[1:]
        folder = resolve_suite_folder(suite_id)
        if folder is None:
            self.send_error(404, "Suite not found")
            return
        if not rel_parts:  # /api/suites/<id> without a trailing slash: relative links need the slash
            self.send_response(302)
            self.send_header("Location", f"/api/suites/{suite_id}/")
            self.end_headers()
            return
        if rel_parts == ["download"]:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(folder.rglob("*")):
                    if f.is_file() and not f.name.endswith(".tmp"):
                        zf.write(f, f"{suite_id}/{f.relative_to(folder).as_posix()}")
            self.send_bytes(buf.getvalue(), "application/zip", filename=f"{suite_id}.zip")
            return
        if rel_parts == [""]:
            rel_parts = ["index.html"]
        for part in rel_parts:
            if part in ("", ".", "..") or any(c in part for c in ("\\", "/", "\x00", ":")):
                self.send_error(404, "Not Found")
                return
        target = (folder / "/".join(rel_parts)).resolve()
        ctype = SUITE_FILES.get(target.suffix.lower())
        if ctype is None or folder not in target.parents or not target.is_file() or target.name.endswith(".tmp"):
            self.send_error(404, "Not Found")
            return
        self.send_bytes(target.read_bytes(), ctype[0], filename=target.name if ctype[1] else None,
                        csp=target.suffix.lower() == ".html", csp_value=SUITE_CSP)

    def handle_ratings_view(self, query: dict):
        """GET /api/ratings/view?report=<id> or ?suite=<id>: configurations, sample images and current ratings to rate."""
        report_id, suite_id = (query.get("report") or [""])[0], (query.get("suite") or [""])[0]
        try:
            view = build_ratings_view(report_id, suite_id)
        except FileNotFoundError as e:
            self.send_json({"error": str(e)}, status=404)
            return
        except ValueError as e:
            self.send_json({"error": str(e)}, status=400)
            return
        self.send_json(view)

    def handle_api_ratings(self):
        """POST /api/ratings {ratings: [{video, model, source, input, conf, coverage, duplicates}], report|suite: id}.
        Stores the ratings (a blank entry removes one) and re-renders every report and suite of those videos and models."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self.send_json({"error": "Invalid JSON"}, status=400)
            return
        entries = req.get("ratings")
        if not isinstance(entries, list) or not entries or len(entries) > 2000 or not all(isinstance(e, dict) for e in entries):
            self.send_json({"error": "ratings must be a non-empty list of entries"}, status=400)
            return
        videos = set(get_available_videos()) | {"synthetic"}
        models = set(get_available_models())
        clean = []
        for e in entries:
            try:
                conf = sweep_mod.conf_value(e.get("conf"))
            except ValueError as err:
                self.send_json({"error": str(err)}, status=400)
                return
            if (str(e.get("video")) not in videos or str(e.get("model")) not in models
                    or not RATING_SOURCE_RE.match(str(e.get("source"))) or not RATING_INPUT_RE.match(str(e.get("input")))):
                self.send_json({"error": "Unknown video, model, source resolution or input size in the ratings"}, status=400)
                return
            clean.append({"video": str(e["video"]), "model": str(e["model"]), "source": str(e["source"]), "input": str(e["input"]),
                          "conf": conf, "coverage": e.get("coverage"), "duplicates": e.get("duplicates")})
        store = RatingStore()
        try:
            with ratings_lock:
                changed = store.update_many(clean)
                done = rerender_matching({(e["video"], e["model"]) for e in clean}, store, skip_suites=[active_suite_id()] if active_suite_id() else [])
        except ValueError as err:
            self.send_json({"error": str(err)}, status=400)
            return
        except OSError as err:
            self.send_json({"error": f"Could not save the ratings: {err}"}, status=500)
            return
        self.send_json({"changed": changed, "rerendered": done})

    def handle_report_file(self, path: str):
        """GET /api/reports/<id>/<file> and /api/reports/<id>/download (ZIP of the whole bundle)."""
        # Split before decoding so an encoded slash cannot smuggle extra path segments
        segments = [unquote(seg) for seg in path[len("/api/reports/"):].split("/")]
        report_id, rel = segments[0], "/".join(segments[1:])
        folder = resolve_report_folder(report_id)
        if folder is None:
            self.send_error(404, "Report not found")
            return

        if rel == "download":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(folder.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"{report_id}/{f.relative_to(folder).as_posix()}")
            self.send_bytes(buf.getvalue(), "application/zip", filename=f"{report_id}.zip")
            return

        if rel in REPORT_FILES:
            ctype, attachment = REPORT_FILES[rel]
        elif REPORT_SAMPLE_RE.match(rel):
            ctype, attachment = "image/jpeg", False
        else:
            self.send_error(404, "Not Found")
            return
        target = (folder / rel).resolve()
        if folder not in target.parents or not target.is_file():
            self.send_error(404, "Not Found")
            return
        self.send_bytes(target.read_bytes(), ctype, filename=target.name if attachment else None,
                        csp=rel == "report.html", csp_value=SUITE_CSP)

    def send_bytes(self, data: bytes, content_type: str, filename: Optional[str] = None, csp: bool = False,
                   csp_value: str = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        if csp:  # the report is fully self-contained: no script, no network
            self.send_header("Content-Security-Policy", csp_value)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

def run_live_inference_worker(req):
    env = req.get("env", "docker")
    target = req.get("target", "x86-cpu")
    model_name = req.get("model", "HumanDetection_light_input_640.onnx")
    video_name = req.get("video", "")
    conf = float(req.get("conf", 0.35))
    save_output = bool(req.get("save_output", True))
    max_ram_mb = req.get("max_ram_mb")
    max_vram_mb = req.get("max_vram_mb")

    video_input_rel = f"videos/input/{video_name}" if video_name else ""
    model_rel = f"models/{model_name}"

    out_rel = ""
    if save_output and video_name:
        out_stem = Path(video_name).stem
        out_rel = f"videos/output/annotated_{out_stem}.mp4"

    env_label = "DOCKER CONTAINER (Hardware Simulated)" if env == "docker" else "HOST PYTHON (Local)"
    with state_lock:
        job_state["logs"].append(f"[LIVE INFERENCE] Starting session on '{target}' via {env_label}...")
        job_state["logs"].append(f"[LIVE INFERENCE] Model: {model_name} | Video: {video_name or 'Webcam'}")
        job_state["logs"].append(f"[LIVE INFERENCE] Save Output: {'YES (' + out_rel + ')' if save_output else 'NO (In-Memory Only)'}")

    if env == "docker":
        cmd = [
            "docker", "compose", "-f", str(PROJECT_ROOT / "docker" / "docker-compose.yml"),
            "run", "--rm", "-i", target,
            "python", "scripts/run_live_stream.py",
            "--target", target,
            "--model", model_rel,
            "--video", video_input_rel,
            "--conf", str(conf)
        ]
        if save_output and out_rel:
            cmd.extend(["--save-output", "--output", out_rel])
    else:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_live_stream.py"),
            "--target", target,
            "--model", str(PROJECT_ROOT / model_rel),
            "--video", str(PROJECT_ROOT / video_input_rel),
            "--conf", str(conf)
        ]
        if save_output and out_rel:
            cmd.extend(["--save-output", "--output", str(PROJECT_ROOT / out_rel)])

    if max_ram_mb:
        cmd.extend(["--max-ram-mb", str(max_ram_mb)])
    if max_vram_mb:
        cmd.extend(["--max-vram-mb", str(max_vram_mb)])

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        with live_session.lock:
            live_session.active = True
            live_session.paused = False
            live_session.stop_requested = False
            live_session.process = proc
            live_session.save_output = save_output
            live_session.output_path = out_rel

        def stream_stderr():
            try:
                for line in iter(proc.stderr.readline, b""):
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        with state_lock:
                            job_state["logs"].append(decoded)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
        stderr_thread.start()

        while True:
            with live_session.lock:
                if live_session.stop_requested:
                    try:
                        if proc.stdin:
                            proc.stdin.write(b"stop\n")
                            proc.stdin.flush()
                    except Exception:
                        pass
                    break

            raw_header = proc.stdout.readline()
            if not raw_header:
                break

            header = raw_header.decode("utf-8", errors="ignore").strip()
            if not header:
                continue

            if header.startswith("READY"):
                parts = header.split()
                if len(parts) >= 3:
                    with live_session.lock:
                        live_session.total_frames = int(parts[1])
                        live_session.fps = float(parts[2])
            elif header.startswith("FRAME"):
                # FRAME <idx> <total> <fps> <latency> <det_count> <jpeg_len>
                parts = header.split()
                if len(parts) >= 7:
                    cur_idx = int(parts[1])
                    tot_frames = int(parts[2])
                    fps_val = float(parts[3])
                    lat_val = float(parts[4])
                    det_cnt = int(parts[5])
                    jpeg_len = int(parts[6])

                    jpeg_bytes = proc.stdout.read(jpeg_len)
                    with live_session.lock:
                        live_session.latest_frame_jpeg = jpeg_bytes
                        live_session.current_frame = cur_idx
                        live_session.total_frames = tot_frames
                        live_session.fps = fps_val
                        live_session.latency_ms = lat_val
                        live_session.detections_count = det_cnt
                        # Newer workers append: <host_latency_ms> <simulated 0|1>
                        live_session.host_latency_ms = float(parts[7]) if len(parts) >= 9 else lat_val
                        live_session.simulated = len(parts) >= 9 and parts[8] == "1"
            elif header.startswith("EOF"):
                with state_lock:
                    job_state["logs"].append("[LIVE INFERENCE] Video playback reached end of stream.")
                break
            elif header.startswith("ERROR"):
                with state_lock:
                    job_state["logs"].append(f"[LIVE INFERENCE ERROR] {header[6:]}")
                break

    except Exception as e:
        with state_lock:
            job_state["logs"].append(f"[LIVE INFERENCE ERROR] {e}")
    finally:
        live_session.reset()
        with state_lock:
            job_state["running"] = False
            job_state["logs"].append("[LIVE INFERENCE] Session finished.")

def track_benchmark_line(line: str, model_count: int) -> bool:
    """Updates job_state progress / report ids from harness output. Returns True for pure progress lines.
    Must be called with state_lock held."""
    m = PROGRESS_RE.match(line)
    if m:
        prev = job_state["progress"] or {}
        job_state["progress"] = {
            "done": int(m.group(1)), "total": int(m.group(2)),
            "model_index": prev.get("model_index", 1), "model_count": prev.get("model_count", model_count),
            "model": prev.get("model", ""), "pass": prev.get("pass"),
        }
        return True
    m = SWEEP_PASS_RE.match(line)
    if m:
        prev = job_state["progress"] or {"done": 0, "total": 0, "model_index": 1, "model_count": model_count, "model": ""}
        job_state["progress"] = {**prev, "pass": {"i": int(m.group(1)), "n": int(m.group(2)), "source": m.group(3), "input": m.group(4)}}
        return True
    m = MODEL_LINE_RE.search(line)
    if m:
        job_state["progress"] = {"done": 0, "total": 0, "model_index": int(m.group(1)),
                                 "model_count": int(m.group(2)), "model": m.group(3)}
        return False
    m = REPORT_LINE_RE.search(line)
    if m and m.group(1) not in job_state["report_ids"]:
        job_state["report_ids"].append(m.group(1))
    return False

def run_job_worker(req):
    env = req.get("env", "docker")
    target = req.get("target", "x86-cpu")
    models = req.get("models") or [req.get("model", "HumanDetection_light_input_640.onnx")]
    video = req.get("video", "")
    frames = str(req.get("frames", 50))
    conf = req.get("conf")
    max_ram_mb = req.get("max_ram_mb")
    max_vram_mb = req.get("max_vram_mb")

    model_args = [f"models/{m}" if env == "docker" else str(PROJECT_ROOT / "models" / m) for m in models]
    video_arg = f"videos/input/{video}" if video else ""
    summary_rel = f"results/metrics/app_run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # Named so Stop can remove the container: killing the compose client alone leaves the benchmark running
    container_name = f"crime-detect-benchmark-{time.strftime('%Y%m%d-%H%M%S')}" if env == "docker" else None

    sub_args = [
        "--target", target,
        "--model", *model_args,
        "--frames", frames,
        "--warmup", "3",
        "--summary-json", summary_rel if env == "docker" else str(PROJECT_ROOT / summary_rel),
        *sweep_mod.sweep_cli_args(req.get("sweep")),
    ]
    notes_path = write_notes_file(req["notes"]) if req.get("notes") else None  # deleted in the finally below
    if notes_path:
        sub_args.extend(["--notes-file", notes_path.relative_to(PROJECT_ROOT).as_posix() if env == "docker" else str(notes_path)])
    if video_arg:
        sub_args.extend(["--video", video_arg])
    if conf is not None:
        sub_args.extend(["--conf", str(conf)])
    if max_ram_mb:
        sub_args.extend(["--max-ram-mb", str(max_ram_mb)])
    if max_vram_mb:
        sub_args.extend(["--max-vram-mb", str(max_vram_mb)])

    if env == "docker":
        cmd = build_docker_run_command(
            target=target,
            script_name="run_benchmark.py",
            script_args=sub_args,
            project_root=PROJECT_ROOT,
            max_ram_mb=max_ram_mb,
            container_name=container_name
        )
    else:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_benchmark.py")] + sub_args

    with state_lock:
        job_state["container"] = container_name
        job_state["logs"].append(f"[APP] Environment: {env.upper()}")
        job_state["logs"].append(f"[APP] Command: {' '.join(cmd)}")

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",  # notes may contain non-ASCII text that ends up in the console output
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        with state_lock:
            job_state["process"] = proc

        for line in proc.stdout:
            line = line.rstrip()
            with state_lock:
                if track_benchmark_line(line, len(models)):
                    continue  # progress ticks only drive the progress bar
                job_state["logs"].append(line)

        proc.wait()

        summary_path = PROJECT_ROOT / summary_rel
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                with state_lock:
                    job_state["metrics"] = metrics_data
            except Exception:
                pass

        with state_lock:
            job_state["logs"].append(f"[APP] Finished with code {proc.returncode}")
    except Exception as e:
        with state_lock:
            job_state["logs"].append(f"[APP ERROR] {e}")
    finally:
        remove_notes_file(notes_path)
        with state_lock:
            # After a Stop the next job may already own job_state: only reset our own run
            if job_state["process"] is proc:
                job_state["running"] = False
                job_state["process"] = None
                job_state["container"] = None

def handle_suite_line(state: dict, line: str) -> bool:
    """Applies a SUITE_* / PROGRESS line of the suite runner to its state. True if the line is machine output
    that should not appear in the console. Must be called with state_lock held."""
    m = PROGRESS_RE.match(line)
    if m:
        if state["current"]:
            state["current"]["done"], state["current"]["total"] = int(m.group(1)), int(m.group(2))
        return True
    m = SUITE_START_RE.match(line)
    if m:
        state["suite_id"], state["total"], state["status"] = m.group(1), int(m.group(2)), "running"
        sync_suite_cells(state, REPORTS_DIR / m.group(1))  # resumed suites keep the results of their finished cells
        job_state["logs"].append(f"[SUITE] {m.group(1)}: {m.group(2)} run(s) to do")
        return True
    m = SWEEP_PASS_RE.match(line)
    if m:
        if state["current"]:
            state["current"]["pass"] = {"i": int(m.group(1)), "n": int(m.group(2)), "source": m.group(3), "input": m.group(4)}
        return True
    if line.startswith("SUITE_TARGET "):
        state["target"] = line.split(" ", 1)[1].strip()
        state["current"] = {"target": state["target"], "model": None, "done": 0, "total": 0}
        job_state["logs"].append(f"[SUITE] starting device {state['target']}")
        return True
    if line.startswith("SUITE_CONTAINER "):
        job_state["container"] = line.split(" ", 1)[1].strip()
        return True
    m = SUITE_CURRENT_RE.match(line)
    if m:
        target, model = m.group(1), m.group(2)
        state["current"] = {"target": target, "model": model, "done": 0, "total": 0}
        if model in state["cells"] and target in state["cells"][model]:
            state["cells"][model][target] = {"status": "running"}
        return True
    m = SUITE_CELL_RE.match(line)
    if m:
        target, model, status = m.group(3), m.group(4), m.group(5)
        extra = dict(tok.split("=", 1) for tok in m.group(6).split())
        state["done"] = int(m.group(1))
        if model in state["cells"] and target in state["cells"][model]:
            fps = float(extra["fps"]) if "fps" in extra else None
            state["cells"][model][target] = {"status": status, "fps": fps, "heat": extra.get("heat")}
        if "dur" in extra:
            state["cell_seconds"].append(float(extra["dur"]))
        job_state["logs"].append(f"[SUITE] {m.group(1)}/{m.group(2)} {target} / {model}: {status}"
                                 + (f" ({extra['fps']} FPS)" if "fps" in extra else ""))
        return True
    if line.startswith("SUITE_DONE"):
        state["done_seen"] = True
        return True
    return False

def heat_of(cell: dict) -> Optional[str]:
    return suite_heat(cell)

def sync_suite_cells(state: dict, folder: Path):
    """Copies the cell statuses (and FPS / colour) of the manifest into the live state."""
    try:
        manifest = load_manifest(folder)
    except (OSError, ValueError, KeyError):
        return
    for model, row in manifest["cells"].items():
        for target, c in row.items():
            if model in state["cells"] and target in state["cells"][model]:
                status = c.get("status", "pending")
                state["cells"][model][target] = {"status": "pending" if status == "running" else status,
                                                 "fps": c.get("fps"), "heat": heat_of(c)}

def run_suite_worker(settings: dict, state: dict):
    env = settings["env"]
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_benchmark_matrix.py")]
    if settings.get("resume"):
        cmd += ["--resume", settings["resume"]]
    else:
        cmd += ["--targets", *settings["targets"], "--models", *[f"models/{m}" for m in settings["models"]],
                "--frames", str(settings["frames"]), "--warmup", str(settings["warmup"])]
        if settings["video"]:
            cmd += ["--video", f"videos/input/{settings['video']}"]
        if settings["conf"] is not None:
            cmd += ["--conf", str(settings["conf"])]
        sw = settings.get("sweep") or {"enabled": False}
        if sw.get("enabled") is False:
            cmd.append("--no-sweep")
        else:
            cmd += ["--source-heights", *["native" if h == 0 else str(h) for h in sw["source_heights"]],
                    "--input-sizes", *([str(x) for x in sw["input_sizes"]] or ["default"]),
                    "--conf-thresholds", *[sweep_mod.conf_text(c) for c in sw["conf_thresholds"]]]
        if env == "local":
            cmd.append("--local")
    notes_path = write_notes_file(settings["notes"]) if settings.get("notes") else None  # read once at start, deleted in the finally below
    if notes_path:
        cmd += ["--notes-file", str(notes_path)]
    with state_lock:
        job_state["container"] = None
        job_state["logs"].append(f"[APP] Benchmark suite ({env.upper()}): {len(state['models'])} model(s) x {len(state['targets'])} device(s)")
        job_state["logs"].append(f"[APP] Command: {' '.join(cmd)}")

    proc = None
    try:
        proc_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, cwd=str(PROJECT_ROOT), env=proc_env)
        with state_lock:
            job_state["process"] = proc
            stopping = state["stopping"]
        if stopping:  # Stop was pressed before the runner existed
            proc.terminate()
        for line in proc.stdout:
            line = line.rstrip()
            with state_lock:
                if handle_suite_line(state, line):
                    continue
                job_state["logs"].append(line)
                if len(job_state["logs"]) > SUITE_LOG_LINES:
                    del job_state["logs"][:-SUITE_LOG_LINES]
        proc.wait()
    except Exception as e:
        with state_lock:
            job_state["logs"].append(f"[APP ERROR] {e}")
    finally:
        remove_notes_file(notes_path)
        with state_lock:
            container = job_state["container"]
            suite_id = state["suite_id"]
            unfinished = not state.get("done_seen") or state["stopping"]
        if env == "docker" and container:  # the runner may have been killed before it could remove its container
            try:
                subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            except Exception:
                pass
        final = "failed"
        if suite_id:
            if unfinished:  # runner killed (Stop) or crashed: mark the rest cancelled and still render the report
                try:
                    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_benchmark_matrix.py"),
                                    "--render-only", suite_id, "--mark-stopped"], cwd=str(PROJECT_ROOT),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
                except Exception:
                    pass
            try:
                final = load_manifest(REPORTS_DIR / suite_id)["status"]
            except (OSError, ValueError, KeyError):
                pass
        with state_lock:
            state["status"] = final
            state["ended_ts"] = time.time()
            state["current"] = None
            if suite_id:
                sync_suite_cells(state, REPORTS_DIR / suite_id)
            for row in state["cells"].values():
                for c in row.values():
                    if c.get("status") == "running":
                        c["status"] = "cancelled"
            job_state["logs"].append(f"[APP] Suite {suite_id or ''} finished: {final}")
            if job_state["suite"] is state:
                job_state["running"] = False
                job_state["process"] = None
                job_state["container"] = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Inference Benchmark — Edge Hardware Simulator & Live Player</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #22c55e;
      --danger: #ef4444;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
      padding: 20px;
      line-height: 1.5;
    }
    .container { max-width: 1240px; margin: 0 auto; }
    header {
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 {
      font-size: 1.5rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      background: #0369a1;
      color: #e0f2fe;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: 370px 1fr;
      gap: 20px;
    }
    .grid > * { min-width: 0; }  /* wide tables scroll inside their card instead of stretching the page */
    @media (max-width: 950px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      color: var(--primary);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .form-group { margin-bottom: 12px; }
    label {
      display: block;
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    select, input {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px;
      color: var(--text);
      font-size: 0.88rem;
      outline: none;
      transition: border-color 0.2s;
    }
    select:focus, input:focus { border-color: var(--primary); }
    .env-toggle {
      display: flex;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 12px;
    }
    .env-btn {
      flex: 1;
      padding: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      text-align: center;
      cursor: pointer;
      color: var(--text-muted);
      background: transparent;
      border: none;
    }
    .env-btn.active {
      background: #0284c7;
      color: #ffffff;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .target-desc {
      background: #0f172a;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 0.78rem;
      color: #cbd5e1;
      margin-top: 4px;
      border-left: 3px solid var(--primary);
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 12px 0;
      background: #0f172a;
      padding: 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .checkbox-row input {
      width: 16px;
      height: 16px;
      cursor: pointer;
    }
    .checkbox-row label {
      margin: 0;
      cursor: pointer;
      color: var(--text);
      font-size: 0.85rem;
    }
    .btn-row {
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }
    button.action-btn {
      flex: 1;
      padding: 11px;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      font-size: 0.92rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-primary { background: var(--primary); color: #0f172a; }
    .btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
    .btn-danger { background: var(--danger); color: #fff; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Video Player */
    .player-container {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .video-viewport {
      width: 100%;
      height: 440px;
      background: #000;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .video-viewport img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .player-controls {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .timeline-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .timeline-slider {
      flex: 1;
      cursor: pointer;
      height: 6px;
    }
    .time-lbl {
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: var(--primary);
      min-width: 105px;
      text-align: right;
    }
    .btn-controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .seek-btns {
      display: flex;
      gap: 6px;
    }
    .ctrl-btn {
      background: #1e293b;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      font-weight: 500;
    }
    .ctrl-btn:hover { background: #334155; }
    .ctrl-btn.play {
      background: var(--primary);
      color: #0f172a;
      font-weight: 700;
    }
    .ctrl-btn.play:hover { background: var(--primary-hover); }

    /* Terminal & KPI */
    .terminal {
      height: 180px;
      background: #090d16;
      border-radius: 8px;
      border: 1px solid var(--border);
      padding: 12px;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #38bdf8;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .kpi-card {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .kpi-val { font-size: 1.2rem; font-weight: 700; color: var(--accent); }
    .kpi-lbl { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }
    .results-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      margin-bottom: 12px;
      font-family: var(--font-mono);
    }
    .results-table th, .results-table td {
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }
    .results-table th:first-child, .results-table td:first-child { text-align: left; }
    .results-table th { color: var(--text-muted); font-weight: 600; font-size: 0.72rem; }
    .results-table td.good { color: var(--accent); }
    .results-table td.warn { color: #f59e0b; }
    .results-table td.bad { color: var(--danger); }
    .table-wrap { overflow-x: auto; }
    .table-note { font-size: 0.72rem; color: var(--text-muted); margin: -6px 0 12px; }
    .btn-secondary {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border) !important;
    }
    .btn-secondary:hover:not(:disabled) { background: #334155; }
    .btn-bench { background: linear-gradient(135deg, #22c55e, #16a34a); color: #04210f; }
    .btn-bench:hover:not(:disabled) { background: #15803d; color: #fff; }
    .progress-track { height: 12px; background: #0f172a; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }
    .progress-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.4s; }
    .progress-fill.indeterminate { width: 30%; animation: slide 1.2s infinite ease-in-out; }
    @keyframes slide { 0% { margin-left: -30%; } 100% { margin-left: 100%; } }
    .progress-text { font-size: 0.8rem; color: var(--text-muted); margin: 6px 0 4px; font-family: var(--font-mono); }
    .report-card { background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-top: 12px; }
    .rc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
    .rc-title { font-weight: 600; font-size: 0.98rem; }
    .rc-sub { font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); word-break: break-all; }
    .vbadge { padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em; }
    .vbadge.ok { background: #14361f; color: #4ade80; }
    .vbadge.warn { background: #3b2f0d; color: #fbbf24; }
    .vbadge.fail { background: #3f1616; color: #f87171; }
    .rc-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 10px; }
    .rc-kpis .kpi-val { font-size: 1.05rem; color: var(--text); }
    .rc-kpis .kpi-val.good { color: var(--accent); }
    .kpi-val.warn { color: #f59e0b; }
    .kpi-val.bad { color: var(--danger); }
    .rc-verdict { list-style: none; font-size: 0.8rem; margin-bottom: 12px; }
    .rc-verdict li { padding: 3px 0; display: flex; gap: 8px; }
    .rc-verdict .pill { flex: none; min-width: 42px; text-align: center; font-size: 0.66rem; font-weight: 700; padding: 1px 6px; border-radius: 5px; height: 18px; }
    .pill.ok { background: #14361f; color: #4ade80; }
    .pill.warn { background: #3b2f0d; color: #fbbf24; }
    .pill.fail { background: #3f1616; color: #f87171; }
    .rc-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    a.link-btn { text-decoration: none; display: inline-block; }
    a.link-btn.play { background: var(--primary); color: #0f172a; }
    .filter-row { display: grid; grid-template-columns: 1fr 2fr; gap: 10px; margin-bottom: 10px; }
    .hist-table th:nth-child(2), .hist-table td:nth-child(2), .hist-table th:nth-child(3), .hist-table td:nth-child(3) { text-align: left; }
    .hist-links a { color: var(--primary); text-decoration: none; margin-right: 8px; font-size: 0.75rem; }
    .hist-links a:hover { text-decoration: underline; }
    .btn-suite { background: linear-gradient(135deg, #a78bfa, #7c3aed); color: #fff; }
    .btn-suite:hover:not(:disabled) { background: #6d28d9; }
    .suite-checks { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 5px 14px; max-height: 230px; overflow: auto; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px; }
    .suite-checks label { display: flex; gap: 8px; align-items: flex-start; margin: 0; color: var(--text); font-size: 0.8rem; cursor: pointer; }
    .suite-checks input { width: auto; margin-top: 3px; }
    .suite-checks .flag { display: block; color: #fbbf24; font-size: 0.7rem; }
    .suite-links a { color: var(--primary); font-size: 0.75rem; cursor: pointer; margin-left: 10px; text-decoration: none; }
    .suite-eta { font-size: 0.82rem; color: #cbd5e1; background: #0f172a; border-left: 3px solid #a78bfa; border-radius: 6px; padding: 8px 10px; margin: 10px 0; }
    .suite-grid-wrap { overflow-x: auto; margin-top: 10px; }
    table.suite-grid { border-collapse: separate; border-spacing: 2px; font-size: 0.72rem; font-family: var(--font-mono); }
    table.suite-grid th { color: var(--text-muted); font-weight: 600; padding: 3px 6px; white-space: nowrap; text-align: center; }
    table.suite-grid th.m { text-align: left; }
    table.suite-grid td { min-width: 68px; text-align: center; padding: 5px 6px; border-radius: 5px; background: #0f172a; color: #64748b; }
    table.suite-grid td.sg-green { background: #14361f; color: #4ade80; }
    table.suite-grid td.sg-amber { background: #3b2f0d; color: #fbbf24; }
    table.suite-grid td.sg-red { background: #3f1616; color: #f87171; }
    table.suite-grid td.sg-failed { background: #334155; color: #f87171; font-weight: 700; }
    table.suite-grid td.sg-cancelled { background: #1e293b; color: #94a3b8; }
    table.suite-grid td.sg-running { background: #0c4a6e; color: #e0f2fe; animation: pulse 1.2s infinite; }
    @keyframes pulse { 50% { opacity: 0.55; } }
    .suite-legend { font-size: 0.7rem; color: var(--text-muted); margin-top: 6px; }
    .notes-box { margin-top: 12px; background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
    .notes-box > summary { cursor: pointer; font-size: 0.85rem; font-weight: 600; }
    .notes-box textarea { width: 100%; background: #0b1220; border: 1px solid var(--border); border-radius: 6px; padding: 7px; color: var(--text); font-size: 0.82rem; font-family: inherit; resize: vertical; min-height: 42px; outline: none; }
    .notes-box textarea:focus { border-color: var(--primary); }
    .note-row { margin: 8px 0; }
    .note-lbl { display: flex; justify-content: space-between; gap: 8px; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 3px; word-break: break-all; }
    .note-lbl a { color: var(--primary); cursor: pointer; flex: none; }
    .note-count { font-size: 0.68rem; color: var(--text-muted); text-align: right; }
    .note-count.full { color: #fbbf24; }
    .note-group-title { font-size: 0.78rem; font-weight: 600; color: var(--primary); margin: 12px 0 2px; border-bottom: 1px solid var(--border); padding-bottom: 2px; }
    #suiteNotesRows { max-height: 380px; overflow: auto; padding-right: 6px; }
    #suiteNotesRows .note-row { display: grid; grid-template-columns: minmax(150px, 34%) 1fr; gap: 4px 12px; align-items: start; }
    #suiteNotesRows .note-lbl { display: block; margin: 0; padding-top: 6px; }
    #suiteNotesRows .note-lbl a { margin-left: 8px; }
    #suiteNotesRows .note-count { grid-column: 2; margin-top: -2px; }
    .results-table td.note-cell, .results-table th.note-th { text-align: left; white-space: pre-wrap; font-family: inherit; min-width: 140px; max-width: 340px; overflow-wrap: anywhere; unicode-bidi: plaintext; }
    .rc-notes { display: flex; gap: 10px; align-items: baseline; font-size: 0.8rem; margin: 0 0 10px; }
    .rc-notes-lbl { flex: none; color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; }
    .rc-notes-val { white-space: pre-wrap; overflow-wrap: anywhere; min-width: 40px; unicode-bidi: plaintext; }
    .sweep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(78px, 1fr)); gap: 4px 8px; margin: 4px 0 6px; }
    .sweep-grid label { display: flex; gap: 6px; align-items: center; margin: 0; color: var(--text); font-size: 0.8rem; cursor: pointer; }
    .sweep-grid input { width: auto; }
    .sweep-grp { font-size: 0.75rem; color: var(--text-muted); margin-top: 8px; }
    .sweep-body input[type="text"] { padding: 6px 8px; font-size: 0.82rem; }
    .rc-best { background: #14221a; border-left: 3px solid #22c55e; border-radius: 6px; padding: 7px 10px; margin: 0 0 10px; font-size: 0.82rem; }
    .rc-best .rc-sub { font-family: inherit; margin-top: 2px; }
    .modal { position: fixed; inset: 0; background: rgba(2, 6, 23, 0.85); z-index: 50; display: none; overflow: auto; padding: 22px 14px; }
    .modal-box { max-width: 1240px; margin: 0 auto; background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .rate-help { font-size: 0.82rem; color: #cbd5e1; background: #0f172a; border-left: 3px solid var(--primary); border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; }
    .rate-item { margin-bottom: 22px; }
    .rate-item h3 { font-size: 1rem; margin-bottom: 2px; }
    .rate-table td.rate-cfg { text-align: left; white-space: normal; min-width: 190px; font-family: inherit; }
    .rate-table td.rate-thumbs { text-align: left; white-space: nowrap; }
    .rate-table td.rate-thumbs img { width: 140px; margin: 0 4px 0 0; border-radius: 4px; border: 1px solid var(--border); cursor: zoom-in; vertical-align: top; }
    .rate-table input.rate-in { width: 70px; padding: 5px 6px; text-align: right; }
    .rate-table tr.rate-best td { background: #14221a; }
    .rate-msg { font-size: 0.85rem; margin: 8px 0; }
    #lightbox { position: fixed; inset: 0; z-index: 60; background: rgba(0, 0, 0, 0.93); display: none; align-items: center; justify-content: center; cursor: zoom-out; }
    #lightbox img { max-width: 96vw; max-height: 94vh; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #64748b; }
    .dot.running { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        <span>Model Inference Benchmark</span>
        <span class="badge">Live Hardware Simulator</span>
      </h1>
      <div style="display: flex; gap: 14px; align-items: center;">
        <span id="dockerBadge" style="font-size: 0.8rem; color: #38bdf8;">🐳 Docker Ready</span>
        <div style="display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: var(--text-muted);">
          <div id="statusDot" class="dot"></div>
          <span id="statusText">Ready</span>
        </div>
      </div>
    </header>

    <div class="grid">
      <!-- Options Sidebar -->
      <div class="card">
        <div class="card-title">Hardware & Pipeline</div>

        <label>Execution Mode</label>
        <div class="env-toggle">
          <button type="button" id="modeInferBtn" class="env-btn active" onclick="setMode('inference')">🎥 Live Inference</button>
          <button type="button" id="modeBenchBtn" class="env-btn" onclick="setMode('benchmark')">📊 Benchmark</button>
        </div>

        <label>Execution Environment</label>
        <div class="env-toggle">
          <button type="button" id="envDockerBtn" class="env-btn active" onclick="setEnv('docker')">🐳 Docker (Hardware Simulated)</button>
          <button type="button" id="envLocalBtn" class="env-btn" onclick="setEnv('local')">💻 Host Python</button>
        </div>

        <div class="form-group">
          <label>Target Hardware Platform</label>
          <select id="targetSelect" onchange="updateTargetInfo(); rebuildNotes()"></select>
          <div id="targetDesc" class="target-desc">Loading details...</div>
        </div>

        <div class="form-group">
          <label id="modelLabel">Model Weights</label>
          <select id="modelSelect" onchange="rebuildNotes()"></select>
        </div>

        <div class="form-group">
          <label>Video Input (videos/input/)</label>
          <select id="videoSelect"></select>
        </div>

        <div class="checkbox-row" id="saveCheckboxContainer">
          <input type="checkbox" id="saveOutputCheck" checked>
          <label for="saveOutputCheck">💾 Save Annotated Video to Disk (.mp4)</label>
        </div>

        <div class="row">
          <div class="form-group">
            <label>RAM Limit (MB)</label>
            <input type="number" id="ramInput" placeholder="e.g. 2048">
          </div>
          <div class="form-group">
            <label>VRAM Limit (MB)</label>
            <input type="number" id="vramInput" placeholder="e.g. 1024">
          </div>
        </div>

        <div class="row">
          <div class="form-group">
            <label>Conf Threshold</label>
            <input type="number" step="0.05" id="confInput" value="0.35" oninput="updateSweepSummaries()">
          </div>
          <div class="form-group">
            <label>Frames (Benchmark)</label>
            <input type="number" id="framesInput" value="50" oninput="updateSweepSummaries()">
          </div>
        </div>

        <div class="btn-row">
          <button id="runBtn" class="action-btn btn-primary" onclick="startRun()">Start Session</button>
          <button id="stopBtn" class="action-btn btn-danger" onclick="stopRun()" disabled>Stop</button>
        </div>
        <div class="btn-row">
          <button id="benchBtn" class="action-btn btn-bench" onclick="startBenchmark()" title="Benchmarks the selected model(s) on the selected target and writes an exportable report">📊 Benchmark &amp; Export Report</button>
        </div>
        <details id="notesBox" class="notes-box">
          <summary>📝 Notes for this benchmark (optional)</summary>
          <div class="table-note" style="margin: 8px 0 0;">One note per model and device. It appears as the Notes column of the report (HTML, JSON, CSV, summary) and in Past Reports. Kept in this browser for the next run.</div>
          <div id="notesRows"></div>
        </details>
        <details id="sweepBox" class="notes-box" open>
          <summary>&#9881; Sweep: resolution &amp; confidence (benchmark)</summary>
          <div class="table-note" style="margin: 8px 0 0;">Every exportable benchmark tests the model over these settings and reports the best configuration. Confidence needs no extra inference.</div>
          <div id="bsSweep" class="sweep-body"></div>
        </details>
        <div class="btn-row">
          <button id="suiteBtn" class="action-btn btn-suite" onclick="openSuitePanel()" title="Runs every selected model on every selected simulated device and builds a multipage benchmark report">🧮 Benchmark All Scenarios</button>
        </div>
        <div class="btn-row">
          <button id="estimateBtn" class="action-btn btn-secondary" onclick="runEstimates()">📐 Estimate All Models × Targets</button>
        </div>
      </div>

      <!-- Main Display Area -->
      <div class="card player-container">
        <!-- Live Video Player (Inference Mode) -->
        <div id="playerSection">
          <div class="card-title">
            <span>Live Detection Feed</span>
            <span id="telemetryBadge" style="font-size: 0.8rem; color: var(--accent); font-family: var(--font-mono);">0.0 FPS | 0 ms</span>
          </div>

          <div id="videoViewport" class="video-viewport">
            <div style="color: #64748b; font-size: 0.9rem;">Click 'Start Session' to begin live detection stream</div>
          </div>

          <div class="player-controls">
            <!-- Timeline scrub slider -->
            <div class="timeline-row">
              <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="1000" value="0" oninput="onSeekChange(this.value)">
              <div id="timeLabel" class="time-lbl">00:00 / 00:00</div>
            </div>

            <!-- Playback Controls -->
            <div class="btn-controls-row">
              <div class="seek-btns">
                <button class="ctrl-btn" onclick="seekOffset(-10)">⏪ -10s</button>
                <button class="ctrl-btn" onclick="seekOffset(-2)">⏪ -2s</button>
                <button id="playPauseBtn" class="ctrl-btn play" onclick="togglePlayPause()">⏸ Pause</button>
                <button class="ctrl-btn" onclick="seekOffset(2)">+2s ⏩</button>
                <button class="ctrl-btn" onclick="seekOffset(10)">+10s ⏩</button>
              </div>
              <div style="font-size: 0.82rem; color: var(--text-muted);">
                Frame: <span id="frameCounter">0 / 0</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Benchmark / Console Section -->
        <div id="consoleSection">
          <div class="card-title">
            <span>Execution Console & KPI Metrics</span>
          </div>

          <div class="kpi-row" id="kpiRow" style="display: none;">
            <div class="kpi-card">
              <div id="kpiFps" class="kpi-val">0.0</div>
              <div id="kpiFpsLbl" class="kpi-lbl">FPS</div>
            </div>
            <div class="kpi-card">
              <div id="kpiLatency" class="kpi-val">0.0 ms</div>
              <div id="kpiLatencyLbl" class="kpi-lbl">Latency (P50)</div>
            </div>
            <div class="kpi-card">
              <div id="kpiRam" class="kpi-val">0 MB</div>
              <div class="kpi-lbl">Peak RAM</div>
            </div>
            <div class="kpi-card">
              <div id="kpiCpu" class="kpi-val">0 %</div>
              <div class="kpi-lbl">Avg CPU Load</div>
            </div>
          </div>

          <div id="resultsTable" class="table-wrap"></div>
          <div id="estimatesTable" class="table-wrap"></div>

          <div id="terminal" class="terminal">[System ready. Choose your model and click 'Start Session']</div>
        </div>
      </div>
    </div>

    <!-- Benchmark suite: setup, live progress + grid, result card (own containers, never touched by log polling) -->
    <div class="card" id="suiteSection" style="margin-top: 20px; display: none;">
      <div class="card-title"><span>🧮 Benchmark All Scenarios</span></div>

      <div id="suiteSetup" style="display: none;">
        <div class="form-group">
          <label>Models <span class="suite-links"><a onclick="suiteSelect('suite-model', true)">all</a><a onclick="suiteSelect('suite-model', false)">none</a></span></label>
          <div id="suiteModels" class="suite-checks"></div>
        </div>
        <div class="form-group">
          <label>Devices (targets.yaml) <span class="suite-links"><a onclick="suiteSelect('suite-device', true)">all</a><a onclick="suiteSelect('suite-device', false)">none</a></span></label>
          <div id="suiteDevices" class="suite-checks"></div>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Frames per run <span class="suite-links"><a onclick="suiteQuick()">Quick (10 frames)</a></span></label>
            <input type="number" id="suiteFrames" value="30" min="1" max="2000" oninput="updateSuiteEta()">
          </div>
          <div class="form-group">
            <label>Warmup iterations</label>
            <input type="number" id="suiteWarmup" value="3" min="0" max="50" oninput="updateSuiteEta()">
          </div>
        </div>
        <div class="row">
          <div class="form-group">
            <label>Video (videos/input/)</label>
            <select id="suiteVideo"></select>
          </div>
          <div class="form-group">
            <label>Conf threshold</label>
            <input type="number" step="0.05" id="suiteConf" value="0.35">
          </div>
        </div>
        <details id="suiteNotesBox" class="notes-box" style="margin: 0 0 10px;">
          <summary>📝 Notes per model/device pair (optional) - <span id="suiteNotesCount">0 of 0 filled</span></summary>
          <div class="table-note" style="margin: 8px 0;">Each note shows up as the Notes column of that pair in the suite report (device and model pages, overview tooltips, results.csv / results.json). Pairs follow the checked models and devices; typed text is kept.</div>
          <input type="text" id="suiteNotesFilter" placeholder="Filter by model, device or note text" oninput="applySuiteNotesFilter()" style="margin-bottom: 4px;">
          <div id="suiteNotesRows"></div>
        </details>
        <details id="suiteSweepBox" class="notes-box" style="margin: 0 0 10px;" open>
          <summary>&#9881; Sweep: source resolution x model input size x confidence</summary>
          <div class="table-note" style="margin: 8px 0 0;">Each model/device pair is tested over these settings; the overview shows the best configuration per pair and why it was picked. Models with a fixed input size ignore the input sizes.</div>
          <div id="ssSweep" class="sweep-body"></div>
        </details>
        <div id="suiteEta" class="suite-eta">-</div>
        <div class="table-note" style="margin: 0 0 8px;">Uses the Docker / Host Python choice on the left. Runs are sequential (one container per device) so timings are not disturbed: do not run live sessions during a suite.</div>
        <div id="suiteMsg" class="table-note" style="color: var(--danger)"></div>
        <div class="btn-row">
          <button id="suiteStartBtn" class="action-btn btn-suite" onclick="startSuite()">▶ Start suite</button>
          <button class="action-btn btn-secondary" onclick="closeSuitePanel()">Cancel</button>
        </div>
      </div>

      <div id="suiteRun" style="display: none;">
        <div id="suiteProgressBox">
          <div class="progress-track"><div id="suiteProgressFill" class="progress-fill indeterminate"></div></div>
          <div id="suiteProgressText" class="progress-text">Starting...</div>
          <div id="suiteCurrent" class="progress-text" style="color: var(--primary);"></div>
          <div class="btn-row" style="margin-top: 8px;"><button class="action-btn btn-danger" style="flex: none; padding: 8px 18px;" onclick="stopRun()">Stop suite</button></div>
        </div>
        <div id="suiteGrid" class="suite-grid-wrap"></div>
        <div class="suite-legend">Cells show estimated FPS (host-measured for x86-cpu): green reaches the required FPS, amber &ge; 5 FPS, red below. &middot; &#9654; running, &#183; pending, &#10005; failed, &ndash; skipped, &oslash; cancelled.</div>
      </div>
      <div id="suiteCards"></div>
    </div>

    <div class="card" id="suiteHistoryCard" style="margin-top: 20px;">
      <div class="card-title">
        <span>Benchmark Suites</span>
        <button type="button" class="ctrl-btn" onclick="loadSuites()">↻ Refresh</button>
      </div>
      <div id="suiteHistory" class="table-wrap"></div>
    </div>

    <!-- Benchmark progress + report cards + history (own containers, never touched by log polling) -->
    <div class="card" id="reportsSection" style="margin-top: 20px;">
      <div class="card-title"><span>Benchmark Reports</span></div>
      <div id="benchProgress" style="display: none;">
        <div class="progress-track"><div id="benchProgressFill" class="progress-fill indeterminate"></div></div>
        <div id="benchProgressText" class="progress-text">Starting...</div>
      </div>
      <div id="benchErrors"></div>
      <div id="reportCards"></div>
      <div id="reportsHint" class="table-note" style="margin: 0;">Press "Benchmark &amp; Export Report" to measure the selected model on the selected target. The report (HTML, JSON, CSV, summary, sample frames) appears here and can be downloaded as a ZIP.</div>
    </div>

    <div class="card" id="historyCard" style="margin-top: 20px;">
      <div class="card-title">
        <span>Past Reports</span>
        <button type="button" class="ctrl-btn" onclick="loadReports()">↻ Refresh</button>
      </div>
      <div class="filter-row">
        <select id="histTarget" onchange="renderHistory()"><option value="">All targets</option></select>
        <input type="text" id="histModel" placeholder="Filter by model name" oninput="renderHistory()">
      </div>
      <div id="historyTable" class="table-wrap"></div>
    </div>
  </div>

  <div id="ratingsModal" class="modal">
    <div class="modal-box">
      <div class="card-title">
        <span>&#11088; Rate detection quality</span>
        <button type="button" class="ctrl-btn" onclick="closeRatings()">&#10005; Close</button>
      </div>
      <div id="ratingsBody"></div>
    </div>
  </div>
  <div id="lightbox" onclick="closeLightbox()"><img id="lightboxImg" alt="Sample frame"></div>

  <script>
    // Embedded Initial Data injected directly by Python on page load
    const SERVER_DATA = "__INITIAL_DATA_PLACEHOLDER__";

    let globalTargets = {};
    let currentMode = 'inference';
    let pollInterval = null;
    let videoStatusInterval = null;
    let isUserSeeking = false;
    let wasRunning = false;
    let lastMetricsJson = '';
    let pendingProgress = null;
    let allReports = [];
    const renderedReports = new Set();

    function populateUI(data) {
      if (!data) return;
      globalTargets = data.targets || {};

      // 1. Populate Target Dropdown
      const targetSel = document.getElementById('targetSelect');
      targetSel.innerHTML = '';
      for (const [key, val] of Object.entries(globalTargets)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${key} (${val.arch || 'any'})`;
        targetSel.appendChild(opt);
      }
      updateTargetInfo();

      // 2. Populate Models Dropdown
      const modelSel = document.getElementById('modelSelect');
      modelSel.innerHTML = '';
      (data.models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        modelSel.appendChild(opt);
      });
      rebuildNotes();

      // 3. Populate Videos Dropdown
      const videoSel = document.getElementById('videoSelect');
      videoSel.innerHTML = '';
      (data.videos || []).forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        videoSel.appendChild(opt);
      });
      if ((data.videos || []).length === 0) {
        videoSel.innerHTML = '<option value="">No videos in videos/input/ (Place video files here)</option>';
      }
    }

    let currentEnv = 'docker';

    function setEnv(env) {
      currentEnv = env;
      if (env === 'docker') {
        document.getElementById('envDockerBtn').classList.add('active');
        document.getElementById('envLocalBtn').classList.remove('active');
      } else {
        document.getElementById('envLocalBtn').classList.add('active');
        document.getElementById('envDockerBtn').classList.remove('active');
      }
    }

    function setMode(mode) {
      currentMode = mode;
      if (mode === 'inference') {
        document.getElementById('modeInferBtn').classList.add('active');
        document.getElementById('modeBenchBtn').classList.remove('active');
        document.getElementById('playerSection').style.display = 'block';
        document.getElementById('saveCheckboxContainer').style.display = 'flex';
        const sel = document.getElementById('modelSelect');
        sel.multiple = false;
        sel.size = 0;
        document.getElementById('modelLabel').textContent = 'Model Weights';
        document.getElementById('runBtn').style.display = '';
      } else {
        document.getElementById('modeBenchBtn').classList.add('active');
        document.getElementById('modeInferBtn').classList.remove('active');
        document.getElementById('playerSection').style.display = 'none';
        document.getElementById('saveCheckboxContainer').style.display = 'none';
        const sel = document.getElementById('modelSelect');
        sel.multiple = true;
        sel.size = Math.min(8, sel.options.length);
        document.getElementById('modelLabel').textContent = 'Model Weights (Ctrl/Shift-click to compare several)';
        document.getElementById('runBtn').style.display = 'none';  // Benchmark & Export Report is the single benchmark path
      }
      rebuildNotes();
    }

    function esc(v) {
      return String(v ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    }

    // ---- Notes per model/device pair (shared by the single benchmark and the suite panel) --------
    const NOTE_MAX = 2000;
    const NOTE_STORE_KEY = 'crimeDetect.notes.v1';
    let noteStore = {};
    try {
      const saved = JSON.parse(localStorage.getItem(NOTE_STORE_KEY) || '{}');
      if (saved && typeof saved === 'object' && !Array.isArray(saved)) noteStore = saved;
    } catch (e) { noteStore = {}; }

    function noteKey(model, target) { return target + '::' + model; }
    function getNote(model, target) { const v = noteStore[noteKey(model, target)]; return typeof v === 'string' ? v : ''; }
    function saveNotes() { try { localStorage.setItem(NOTE_STORE_KEY, JSON.stringify(noteStore)); } catch (e) {} }
    function noteCountText(len) { return len >= NOTE_MAX ? len + '/' + NOTE_MAX + ' - limit reached, longer text is cut' : len + '/' + NOTE_MAX; }

    function noteRowHtml(model, target, label) {
      return `<div class="note-row" data-model="${esc(model)}" data-target="${esc(target)}">
        <div class="note-lbl"><span>${label}</span><a onclick="clearNote(this)">clear</a></div>
        <textarea class="note-input" maxlength="${NOTE_MAX}" dir="auto" rows="2" placeholder="Optional note" oninput="onNoteInput(this)"></textarea>
        <div class="note-count"></div></div>`;
    }

    function setNoteInput(row, text) {
      const ta = row.querySelector('textarea');
      if (ta.value !== text) ta.value = text;
      const cnt = row.querySelector('.note-count');
      cnt.textContent = text ? noteCountText(text.length) : '';
      cnt.classList.toggle('full', text.length >= NOTE_MAX);
    }

    function fillNoteInputs(container) {
      container.querySelectorAll('.note-row').forEach(row => setNoteInput(row, getNote(row.dataset.model, row.dataset.target)));
    }

    function onNoteInput(ta) {
      const row = ta.closest('.note-row');
      const key = noteKey(row.dataset.model, row.dataset.target);
      if (ta.value.trim() === '') delete noteStore[key]; else noteStore[key] = ta.value;
      saveNotes();
      document.querySelectorAll('.note-row').forEach(r => {  // the same pair may be shown in both panels
        if (noteKey(r.dataset.model, r.dataset.target) === key) setNoteInput(r, ta.value);
      });
      updateSuiteNotesCount();
    }

    function clearNote(link) {
      const ta = link.closest('.note-row').querySelector('textarea');
      ta.value = '';
      onNoteInput(ta);
    }

    function collectNotes(models, targets) {
      const out = {};
      for (const m of models) {
        for (const t of targets) {
          const v = getNote(m, t).trim();
          if (v) { (out[m] = out[m] || {})[t] = v; }
        }
      }
      return out;
    }

    function selectedBenchModels() {
      const sel = document.getElementById('modelSelect');
      let models = Array.from(sel.selectedOptions).map(o => o.value);
      if (models.length === 0 && sel.value) models = [sel.value];
      return models;
    }

    function rebuildNotes() {
      const target = document.getElementById('targetSelect').value;
      const box = document.getElementById('notesRows');
      const models = selectedBenchModels();
      box.innerHTML = models.length
        ? models.map(m => noteRowHtml(m, target, esc(m) + ' on ' + esc(target))).join('')
        : '<div class="table-note">Select a model.</div>';
      fillNoteInputs(box);
      updateSweepSummaries();
    }

    function updateSuiteNotesCount() {
      const el = document.getElementById('suiteNotesCount');
      if (!el) return;
      const rows = Array.from(document.querySelectorAll('#suiteNotesRows .note-row'));
      el.textContent = rows.filter(r => getNote(r.dataset.model, r.dataset.target).trim()).length + ' of ' + rows.length + ' filled';
    }

    function applySuiteNotesFilter() {
      const q = document.getElementById('suiteNotesFilter').value.trim().toLowerCase();
      document.querySelectorAll('#suiteNotesRows .note-row').forEach(r => {
        const hay = (r.dataset.model + ' ' + r.dataset.target + ' ' + r.querySelector('textarea').value).toLowerCase();
        r.style.display = !q || hay.includes(q) ? '' : 'none';
      });
      document.querySelectorAll('#suiteNotesRows .note-group').forEach(g => {
        g.style.display = Array.from(g.querySelectorAll('.note-row')).some(r => r.style.display !== 'none') ? '' : 'none';
      });
    }

    function rebuildSuiteNotes() {
      const models = suiteChecked('suite-model');
      const devs = suiteChecked('suite-device');
      const box = document.getElementById('suiteNotesRows');
      let html = '';
      for (const t of devs) {
        const hw = (globalTargets[t] || {}).hardware || {};
        html += `<div class="note-group"><div class="note-group-title">${esc(t)} <span style="color: var(--text-muted); font-weight: 400;">${esc(hw.device || '')}</span></div>`;
        for (const m of models) html += noteRowHtml(m, t, esc(m) + ' <span style="opacity: 0.7;">&middot; ' + esc(t) + '</span>');
        html += '</div>';
      }
      box.innerHTML = html && models.length ? html : '<div class="table-note">Select at least one model and one device.</div>';
      fillNoteInputs(box);
      applySuiteNotesFilter();
      updateSuiteNotesCount();
    }

    function onSuiteSelection() { updateSuiteEta(); rebuildSuiteNotes(); }

    function hardwareInfo(hw) {
      if (!hw) return '';
      if (hw.simulate === false) {
        return '<br><strong>Timing:</strong> measured on this host (no device simulation)';
      }
      const ref = hw.reference || {};
      return `<br><strong>Simulated device:</strong> ${esc(hw.device)} — ${esc(hw.runtime)}<br>
        <strong>CPU cores:</strong> ${esc(hw.cpu_cores || 'all')} |
        <strong>Anchor:</strong> ${esc(ref.model)} = ${esc(ref.latency_ms)} ms (${esc(ref.gflops)} GFLOPs)`;
    }

    function updateTargetInfo() {
      const selected = document.getElementById('targetSelect').value;
      const info = globalTargets[selected];
      const descEl = document.getElementById('targetDesc');
      if (info) {
        descEl.innerHTML = `
          <strong>Arch:</strong> ${info.arch || 'N/A'} | 
          <strong>Accel:</strong> ${info.accelerator || 'none'}<br>
          <strong>Default RAM:</strong> ${info.ram_limit_mb ? info.ram_limit_mb + ' MB' : 'Unlimited'} |
          <strong>Default VRAM:</strong> ${info.vram_limit_mb ? info.vram_limit_mb + ' MB' : 'N/A'}<br>
          <em>${info.description || ''}</em>
          ${hardwareInfo(info.hardware)}
        `;
        document.getElementById('ramInput').value = info.ram_limit_mb || '';
        document.getElementById('vramInput').value = info.vram_limit_mb || '';
      }
    }

    function benchmarkPayload() {
      const models = selectedBenchModels();
      const target = document.getElementById('targetSelect').value;
      return {
        mode: 'benchmark',
        env: currentEnv,
        target: target,
        model: models[0] || '',
        models: models,
        notes: collectNotes(models, [target]),
        sweep: readSweep('bs'),
        video: document.getElementById('videoSelect').value,
        conf: parseFloat(document.getElementById('confInput').value) || 0.35,
        frames: parseInt(document.getElementById('framesInput').value) || 50,
        max_ram_mb: parseInt(document.getElementById('ramInput').value) || null,
        max_vram_mb: parseInt(document.getElementById('vramInput').value) || null,
      };
    }

    async function startBenchmark() {
      const payload = benchmarkPayload();
      if (payload.models.length === 0) {
        document.getElementById('terminal').textContent = '[APP] No model selected.';
        return;
      }
      resetReportArea();
      document.getElementById('kpiRow').style.display = 'none';
      document.getElementById('resultsTable').innerHTML = '';
      lastMetricsJson = '';
      setRunningButtons(true);
      pendingProgress = { done: 0, total: 0, model_index: 1, model_count: payload.models.length, model: payload.models[0] };
      showProgress(pendingProgress);
      try {
        const res = await fetch('/api/benchmark', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
          document.getElementById('terminal').textContent = '[APP] ' + data.error;
          hideProgress();
          setRunningButtons(false);
          return;
        }
      } catch (err) {
        document.getElementById('terminal').textContent = '[APP] Could not start benchmark: ' + err;
        hideProgress();
        setRunningButtons(false);
        return;
      }
      startPolling();
    }

    function setRunningButtons(running) {
      document.getElementById('runBtn').disabled = running;
      document.getElementById('benchBtn').disabled = running;
      document.getElementById('suiteBtn').disabled = running;
      document.getElementById('stopBtn').disabled = !running;
    }

    async function startRun() {
      if (currentMode === 'benchmark') return startBenchmark();
      const payload = {
        mode: currentMode,
        env: currentEnv,
        target: document.getElementById('targetSelect').value,
        model: document.getElementById('modelSelect').value,
        models: Array.from(document.getElementById('modelSelect').selectedOptions).map(o => o.value),
        video: document.getElementById('videoSelect').value,
        save_output: document.getElementById('saveOutputCheck').checked,
        conf: parseFloat(document.getElementById('confInput').value) || 0.35,
        frames: parseInt(document.getElementById('framesInput').value) || 50,
        max_ram_mb: parseInt(document.getElementById('ramInput').value) || null,
        max_vram_mb: parseInt(document.getElementById('vramInput').value) || null,
      };

      document.getElementById('runBtn').disabled = true;
      document.getElementById('stopBtn').disabled = false;
      document.getElementById('kpiRow').style.display = 'none';
      document.getElementById('resultsTable').innerHTML = '';

      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        document.getElementById('terminal').textContent = '[APP ERROR] ' + (err.error || ('HTTP ' + res.status));
        document.getElementById('runBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        return;
      }

      // Connect the video stream only after the session has been accepted
      if (currentMode === 'inference') connectFeed();

      startPolling();
    }

    let lastFeedConnect = 0;
    function connectFeed() {
      lastFeedConnect = Date.now();
      const vp = document.getElementById('videoViewport');
      vp.innerHTML = '<img id="liveFeedImg" src="/api/video_feed?t=' + Date.now() + '" alt="Live Detection Stream">';
    }

    async function stopRun() {
      await fetch('/api/stop', { method: 'POST' });
    }

    async function togglePlayPause() {
      const res = await fetch('/api/video/playback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'toggle' })
      });
      const data = await res.json();
      document.getElementById('playPauseBtn').textContent = data.paused ? '▶ Play' : '⏸ Pause';
    }

    async function seekOffset(seconds) {
      await fetch('/api/video/seek', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ offset_seconds: seconds })
      });
    }

    async function onSeekChange(val) {
      isUserSeeking = true;
      const pct = val / 10.0;
      await fetch('/api/video/seek', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pct: pct })
      });
      setTimeout(() => { isUserSeeking = false; }, 400);
    }

    function startPolling() {
      if (pollInterval) clearInterval(pollInterval);
      if (videoStatusInterval) clearInterval(videoStatusInterval);

      pollInterval = setInterval(fetchLogs, 1000);
      videoStatusInterval = setInterval(fetchVideoStatus, 500);
    }

    async function fetchVideoStatus() {
      if (currentMode !== 'inference') return;
      try {
        const res = await fetch('/api/video/status');
        const data = await res.json();

        if (data.active) {
          // Session running but no stream open (dropped connection, page reload): reconnect the player
          if (data.viewers === 0 && Date.now() - lastFeedConnect > 3000) connectFeed();
          const simTxt = data.simulated ? ` (est. on device) | host ${data.host_latency_ms} ms` : '';
          document.getElementById('telemetryBadge').textContent = `${data.fps} FPS | ${data.latency_ms} ms${simTxt} | ${data.detections} Dets`;
          document.getElementById('timeLabel').textContent = `${data.current_time} / ${data.total_time}`;
          document.getElementById('frameCounter').textContent = `${data.current_frame} / ${data.total_frames}`;

          if (!isUserSeeking) {
            document.getElementById('timelineSlider').value = Math.round(data.progress_pct * 10);
          }
          document.getElementById('playPauseBtn').textContent = data.paused ? '▶ Play' : '⏸ Pause';
        }
      } catch (err) {}
    }

    async function fetchLogs() {
      try {
        const res = await fetch('/api/logs');
        const data = await res.json();

        const term = document.getElementById('terminal');
        if (data.logs.length > 0) {
          term.textContent = data.logs.join('\\n');
          term.scrollTop = term.scrollHeight;
        }

        const dot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const isBench = data.last_run_type === 'benchmark';

        if (data.running) {
          dot.classList.add('running');
          statusText.textContent = 'Running...';
          setRunningButtons(true);
          if (!pollInterval) startPolling();  // re-attach after a page reload
          if (isBench) showProgress(data.progress);        } else {
          dot.classList.remove('running');
          statusText.textContent = 'Ready';
          setRunningButtons(false);
          hideProgress();

          if (data.metrics) {
            displayMetrics(data.metrics);
          }
        }
        if (isBench) {
          syncReportCards(data.report_ids || []);
          renderRunErrors(data.running ? null : data.metrics);
        }
        if (wasRunning && !data.running && isBench) loadReports();
        if (data.last_run_type === 'suite' && data.suite) renderSuiteRun(data.suite, data.running);
        if (wasRunning && !data.running && data.last_run_type === 'suite') loadSuites();
        wasRunning = data.running;
      } catch (err) {}
    }

    function fpsClass(fps) {
      if (fps === undefined || fps === null || fps === '') return '';
      return fps >= 15 ? 'good' : (fps >= 5 ? 'warn' : 'bad');
    }

    function displayMetrics(data) {
      const json = JSON.stringify(data);
      if (json === lastMetricsJson) return;  // polling calls this every second: only redraw on change
      lastMetricsJson = json;
      const runs = (data.runs || [data]).filter(r => r && !r.error);
      const failed = (data.runs || []).filter(r => r && r.error);
      if (runs.length === 0 && failed.length === 0) return;

      if (runs.length > 0) {
        const m = runs[0];
        const sim = m.simulated;
        document.getElementById('kpiRow').style.display = 'grid';
        document.getElementById('kpiFps').textContent = sim ? sim.est_fps : (m.throughput_fps || '0.0');
        document.getElementById('kpiFpsLbl').textContent = sim ? 'Est. Device FPS' : 'FPS';
        document.getElementById('kpiLatency').textContent = (sim ? sim.latency_ms.p50 : (m.latency_ms?.p50 || 0)) + ' ms';
        document.getElementById('kpiLatencyLbl').textContent = sim ? 'Est. Device Latency (P50)' : 'Latency (P50)';
        document.getElementById('kpiRam').textContent = (m.system_resources?.peak_ram_mb || 0) + ' MB';
        document.getElementById('kpiCpu').textContent = (m.system_resources?.avg_cpu_percent || 0) + ' %';
      }

      const anySim = runs.some(r => r.simulated);
      let html = '<table class="results-table"><tr><th>Model</th><th>GFLOPs</th><th>Host ms</th><th>Host FPS</th>';
      if (anySim) html += '<th>Est. device ms</th><th>Est. device FPS</th>';
      html += '<th>Peak RAM</th><th>Detections</th><th class="note-th">Notes</th></tr>';
      for (const r of runs) {
        const sim = r.simulated || {};
        html += `<tr><td>${esc(r.model)}</td><td>${esc(r.model_profile?.gflops ?? '-')}</td>
          <td>${esc(r.latency_ms?.mean)}</td><td>${esc(r.throughput_fps)}</td>`;
        if (anySim) html += `<td>${esc(sim.latency_ms?.mean ?? '-')}</td><td class="${fpsClass(sim.est_fps)}">${esc(sim.est_fps ?? '-')}</td>`;
        html += `<td>${esc(r.system_resources?.peak_ram_mb)} MB</td><td>${esc(r.total_detections)}</td><td class="note-cell" dir="auto">${esc(r.notes || '')}</td></tr>`;
      }
      for (const r of failed) {
        html += `<tr><td>${esc(r.model)}</td><td class="bad" colspan="${anySim ? 7 : 5}">${esc(r.error)}</td><td class="note-cell" dir="auto">${esc(r.notes || '')}</td></tr>`;
      }
      html += '</table>';
      if (anySim) {
        const dev = runs.find(r => r.simulated).simulated;
        html += `<div class="table-note">Device estimates for ${esc(dev.device)} (${esc(dev.runtime)}), scaled from ${esc(dev.reference.model)}.
          Expect roughly ±30–50% vs. real hardware. Detections and RAM are measured for real.</div>`;
      }
      document.getElementById('resultsTable').innerHTML = html;
    }

    // ---- Benchmark progress, report cards and history -------------------------------------------
    function showProgress(p) {
      document.getElementById('benchProgress').style.display = 'block';
      document.getElementById('reportsHint').style.display = 'none';
      const fill = document.getElementById('benchProgressFill');
      const text = document.getElementById('benchProgressText');
      if (!p || !p.total) {
        p = p || pendingProgress;
        fill.classList.add('indeterminate');
        fill.style.width = '';
        text.textContent = p && p.model ? `Model ${p.model_index}/${p.model_count}: ${p.model} - loading model and warming up...` : 'Starting...';
        return;
      }
      fill.classList.remove('indeterminate');
      const overall = ((p.model_index - 1) + p.done / p.total) / Math.max(1, p.model_count);
      fill.style.width = Math.round(overall * 100) + '%';
      const pass = p.pass ? ` - ${p.pass.source} \u00b7 ${p.pass.input} \u00b7 pass ${p.pass.i}/${p.pass.n}` : '';
      text.textContent = `Model ${p.model_index}/${p.model_count}${p.model ? ' (' + p.model + ')' : ''}${pass} - frame ${p.done}/${p.total} - ${Math.round(overall * 100)}% overall`;
    }

    function hideProgress() {
      document.getElementById('benchProgress').style.display = 'none';
    }

    function resetReportArea() {
      renderedReports.clear();
      document.getElementById('reportCards').innerHTML = '';
      document.getElementById('benchErrors').innerHTML = '';
    }

    function renderRunErrors(metrics) {
      const box = document.getElementById('benchErrors');
      const failed = metrics ? (metrics.runs || []).filter(r => r && r.error) : [];
      box.innerHTML = failed.map(r => `<div class="report-card"><div class="rc-head"><div class="rc-title">${esc(r.model)} on ${esc(r.target)}</div><span class="vbadge fail">FAILED</span></div><div class="rc-sub">${esc(r.error)}</div></div>`).join('');
    }

    function reportLinks(id) {
      const base = '/api/reports/' + encodeURIComponent(id);
      return { zip: base + '/download', html: base + '/report.html', json: base + '/report.json', csv: base + '/frames.csv', md: base + '/summary.md' };
    }

    function reportCardHtml(r) {
      const est = r.device_estimate;
      const rt = r.realtime;
      const links = reportLinks(r.report_id);
      const fpsCls = fpsClass(est ? est.est_fps : r.host_performance.throughput_fps);
      const ramBad = r.resources.ram_limit_breached ? 'bad' : '';
      const tiles = [];
      if (est) {
        tiles.push([Number(est.est_fps).toFixed(1), 'Est. device FPS', fpsCls]);
        tiles.push([est.latency_ms.p50 + ' / ' + est.latency_ms.p95 + ' ms', 'Est. latency P50 / P95', '']);
      } else {
        tiles.push([Number(r.host_performance.throughput_fps).toFixed(1), 'Host FPS (not simulated)', fpsCls]);
      }
      tiles.push([est ? Number(r.host_performance.throughput_fps).toFixed(1) : r.host_performance.latency_ms.p50 + ' ms', est ? 'Host FPS (measured)' : 'Host latency P50', '']);
      tiles.push([r.detections.mean_per_frame, 'Detections / frame', '']);
      tiles.push([r.resources.peak_ram_mb + ' MB' + (r.resources.ram_limit_mb ? ' / ' + r.resources.ram_limit_mb : ''), 'Peak RAM / budget', ramBad]);
      tiles.push([rt.device_realtime_factor + 'x', 'Real-time factor (' + rt.required_fps + ' FPS)', rt.realtime_capable ? 'good' : (rt.device_realtime_factor >= 0.5 ? 'warn' : 'bad')]);
      const tilesHtml = tiles.map(t => `<div class="kpi-card"><div class="kpi-val ${t[2]}">${esc(t[0])}</div><div class="kpi-lbl">${esc(t[1])}</div></div>`).join('');
      const verdictHtml = r.verdict.items.map(i => `<li><span class="pill ${esc(i.level)}">${esc(i.level.toUpperCase())}</span><span>${esc(i.message)}</span></li>`).join('');
      return `<div class="report-card" id="rc-${esc(r.report_id)}">
        <div class="rc-head">
          <div><div class="rc-title">${esc(r.model.file)} on ${esc(r.target.device)}</div>
          <div class="rc-sub">${esc(r.report_id)}</div></div>
          <span class="vbadge ${esc(r.verdict.overall)}">${esc(r.verdict.overall.toUpperCase())}</span>
        </div>
        <div class="rc-notes"><span class="rc-notes-lbl">Notes</span><div class="rc-notes-val" dir="auto">${esc(r.notes || '')}</div></div>
        ${bestConfigHtml(r)}
        <div class="rc-kpis">${tilesHtml}</div>
        <ul class="rc-verdict">${verdictHtml}</ul>
        <div class="rc-actions">
          <a class="ctrl-btn play link-btn" href="${links.zip}">⬇ Download ZIP</a>
          <a class="ctrl-btn link-btn" href="${links.html}" target="_blank" rel="noopener">Open HTML report</a>
          <a class="ctrl-btn link-btn" href="${links.json}" target="_blank" rel="noopener">JSON</a>
          <a class="ctrl-btn link-btn" href="${links.csv}">CSV</a>
          <a class="ctrl-btn link-btn" href="${links.md}">Summary (.md)</a>
          ${r.sweep ? `<button type="button" class="ctrl-btn" data-id="${esc(r.report_id)}" onclick="openRatings('report', this.dataset.id)">&#11088; Rate detection quality</button>` : ''}
        </div>
      </div>`;
    }

    const RULE_NAMES = { user_rating: 'manual rating', auto_realtime_stable: 'real-time + stable (automatic)',
                         auto_fastest: 'fastest, not real-time (automatic)', single_config: 'single configuration' };

    function bestConfigHtml(r) {
      const sw = r.sweep;
      if (!sw || !sw.best) return '';
      const e = (sw.configs || []).find(x => x.id === sw.best.config_id) || {};
      const auto = sw.best.auto_pick ? `<div class="rc-sub">Automatic pick without ratings: ${esc(sw.best.auto_pick.label)}</div>` : '';
      return `<div class="rc-best"><b>Best configuration</b> (${(sw.configs || []).length} tested): ${esc(e.label || sw.best.label)}
        <div class="rc-sub">${esc(RULE_NAMES[sw.best.rule] || sw.best.rule)}: ${esc(sw.best.reason)}</div>${auto}</div>`;
    }

    async function refreshReportCard(id) {
      const el = document.getElementById('rc-' + id);
      if (!el) return;
      try {
        const res = await fetch('/api/reports/' + encodeURIComponent(id) + '/report.json');
        if (res.ok) el.outerHTML = reportCardHtml(await res.json());
      } catch (err) {}
    }

    async function syncReportCards(ids) {
      const box = document.getElementById('reportCards');
      for (const id of ids) {
        if (renderedReports.has(id)) continue;
        renderedReports.add(id);  // marked first: polling may call again before the fetch returns
        try {
          const res = await fetch('/api/reports/' + encodeURIComponent(id) + '/report.json');
          if (!res.ok) throw new Error('HTTP ' + res.status);
          const report = await res.json();
          box.insertAdjacentHTML('beforeend', reportCardHtml(report));
          document.getElementById('reportsHint').style.display = 'none';
        } catch (err) {
          renderedReports.delete(id);
        }
      }
    }

    async function loadReports() {
      try {
        const res = await fetch('/api/reports');
        const data = await res.json();
        allReports = data.reports || [];
        const sel = document.getElementById('histTarget');
        const keep = sel.value;
        const targets = Array.from(new Set(allReports.map(r => r.target))).sort();
        sel.innerHTML = '<option value="">All targets</option>' + targets.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
        sel.value = targets.includes(keep) ? keep : '';
        renderHistory();
      } catch (err) {
        document.getElementById('historyTable').innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(err)}</div>`;
      }
    }

    function renderHistory() {
      const target = document.getElementById('histTarget').value;
      const text = document.getElementById('histModel').value.trim().toLowerCase();
      const rows = allReports.filter(r => (!target || r.target === target) && (!text || r.model.toLowerCase().includes(text)));
      if (rows.length === 0) {
        document.getElementById('historyTable').innerHTML = `<div class="table-note">${allReports.length ? 'No reports match the filter.' : 'No reports yet. Run a benchmark to create one.'}</div>`;
        return;
      }
      let html = '<table class="results-table hist-table"><tr><th>Date</th><th>Target / device</th><th>Model</th><th class="note-th">Notes</th><th>Best configuration</th><th>Est. FPS</th><th>Host FPS</th><th>Dets/frame</th><th>Peak RAM</th><th>Verdict</th><th>Export</th></tr>';
      for (const r of rows) {
        const l = reportLinks(r.report_id);
        const when = new Date(r.created);
        const dateTxt = isNaN(when) ? r.created : when.toLocaleString();
        const verdictCls = r.overall === 'ok' ? 'good' : (r.overall === 'warn' ? 'warn' : 'bad');
        html += `<tr><td>${esc(dateTxt)}</td><td>${esc(r.target)}<br><span style="color: var(--text-muted)">${esc(r.device)}</span></td>
          <td>${esc(r.model)}</td><td class="note-cell" dir="auto">${esc(r.notes || '')}</td>
          <td style="text-align: left; white-space: normal; font-family: inherit;">${r.best_config ? esc(r.best_config.short) + (r.best_config.rated ? ' &#9733;' : '') + '<br><span style="color: var(--text-muted)">' + esc(RULE_NAMES[r.best_config.rule] || r.best_config.rule) + '</span>' : '<span style="color: var(--text-muted)">single configuration</span>'}</td>
          <td class="${r.est_fps === null ? '' : fpsClass(r.est_fps)}">${r.est_fps === null ? 'n/a' : Number(r.est_fps).toFixed(2)}</td>
          <td>${Number(r.host_fps).toFixed(2)}</td><td>${esc(r.detections_per_frame)}</td><td>${esc(r.peak_ram_mb)} MB</td>
          <td class="${verdictCls}">${esc(r.overall.toUpperCase())}</td>
          <td class="hist-links"><a href="${l.zip}">ZIP</a><a href="${l.html}" target="_blank" rel="noopener">HTML</a><a href="${l.json}" target="_blank" rel="noopener">JSON</a><a href="${l.csv}">CSV</a><a href="${l.md}">MD</a>${r.best_config ? `<a href="#" data-id="${esc(r.report_id)}" onclick="openRatings('report', this.dataset.id); return false;">&#11088; Rate</a>` : ''}</td></tr>`;
      }
      html += '</table>';
      document.getElementById('historyTable').innerHTML = html;
    }

    async function runEstimates() {
      const btn = document.getElementById('estimateBtn');
      const box = document.getElementById('estimatesTable');
      btn.disabled = true;
      btn.textContent = 'Estimating...';
      box.innerHTML = '<div class="table-note">Profiling models (no inference is run)...</div>';
      try {
        const res = await fetch('/api/estimates', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ env: currentEnv })
        });
        const data = await res.json();
        if (data.error) {
          box.innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(data.error)}</div>`;
          return;
        }
        const targets = Object.entries(data.targets).filter(([, t]) => t.simulated);
        let html = '<table class="results-table"><tr><th>Model</th><th>GFLOPs</th>';
        for (const [name, t] of targets) html += `<th title="${esc(t.runtime)}">${esc(t.device)}</th>`;
        html += '</tr>';
        for (const row of data.rows) {
          html += `<tr><td title="${esc(row.error || '')}">${esc(row.name)}</td><td>${esc(row.gflops ?? '-')}</td>`;
          for (const [name] of targets) {
            const c = row.estimates[name];
            html += c ? `<td class="${fpsClass(c.fps)}">${c.fps} FPS<br><span style="color: var(--text-muted)">${c.inference_ms} ms</span></td>` : '<td>n/a</td>';
          }
          html += '</tr>';
        }
        html += '</table><div class="table-note">Model inference only, estimated from GFLOPs and the published benchmark anchor of each device. ' +
                'Pre/post-processing on the device CPU comes on top: run a Benchmark for full-pipeline numbers.</div>';
        box.innerHTML = html;
      } catch (err) {
        box.innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(err)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = '📐 Estimate All Models × Targets';
      }
    }

    // ---- Benchmark suite: every model x every device --------------------------------------------
    const SUITE_EST = SERVER_DATA.suite_estimate;
    const RUNNABLE_EXT = ['.onnx', '.pt'];
    let suiteGflops = {};
    let suiteRows = [];
    let suiteCardId = null;
    let suiteSetupOpen = false;
    let lastSuiteSig = '';

    function fmtDur(sec) {
      const s = Math.max(0, Math.round(sec));
      if (s >= 3600) return Math.floor(s / 3600) + 'h ' + String(Math.floor((s % 3600) / 60)).padStart(2, '0') + 'm';
      return s >= 60 ? Math.floor(s / 60) + 'm ' + String(s % 60).padStart(2, '0') + 's' : s + 's';
    }
    function shortModel(m) { const i = m.lastIndexOf('.'); return i > 0 ? m.slice(0, i) : m; }
    function isRunnable(m) { return RUNNABLE_EXT.includes(m.slice(m.lastIndexOf('.')).toLowerCase()); }
    function suiteChecked(cls) { return Array.from(document.querySelectorAll('input.' + cls + ':checked')).map(el => el.value); }
    function suiteSelect(cls, on) { document.querySelectorAll('input.' + cls).forEach(el => { el.checked = on; }); onSuiteSelection(); }
    function suiteQuick() { document.getElementById('suiteFrames').value = 10; updateSuiteEta(); }

    function buildSuiteSetup() {
      const flags = SERVER_DATA.model_flags || {};
      document.getElementById('suiteModels').innerHTML = (SERVER_DATA.models || []).map(m =>
        `<label><input type="checkbox" class="suite-model" value="${esc(m)}" ${flags[m] ? '' : 'checked'} onchange="onSuiteSelection()">
         <span>${esc(m)}${flags[m] ? `<span class="flag">${esc(flags[m])}</span>` : ''}</span></label>`).join('');
      document.getElementById('suiteDevices').innerHTML = Object.entries(globalTargets).map(([key, t]) => {
        const hw = t.hardware || {};
        const sim = hw.simulate === false ? ' (host-measured, not simulated)' : '';
        return `<label><input type="checkbox" class="suite-device" value="${esc(key)}" checked onchange="onSuiteSelection()">
          <span>${esc(key)}<span class="flag" style="color: var(--text-muted)">${esc(hw.device || t.description || '')}${esc(sim)}</span></span></label>`;
      }).join('');
      const vids = ['<option value="">(synthetic frames)</option>'].concat((SERVER_DATA.videos || []).map(v => `<option value="${esc(v)}">${esc(v)}</option>`));
      const sel = document.getElementById('suiteVideo');
      sel.innerHTML = vids.join('');
      sel.value = document.getElementById('videoSelect').value || '';
      document.getElementById('suiteConf').value = document.getElementById('confInput').value;
      updateSuiteEta();
      rebuildSuiteNotes();
    }

    function updateSuiteEta() {
      const models = suiteChecked('suite-model').filter(isRunnable);
      const devs = suiteChecked('suite-device');
      const frames = parseInt(document.getElementById('suiteFrames').value) || 1;
      const warmup = parseInt(document.getElementById('suiteWarmup').value) || 0;
      let perTarget = SUITE_EST.container_start_s;
      const sw = readSweep('ss');
      for (const m of models) perTarget += cellSeconds(m, frames, warmup, sw);
      const runs = models.length * devs.length;
      document.getElementById('suiteEta').textContent = runs === 0
        ? 'Select at least one model and one device.'
        : `${runs} runs (${models.length} models x ${devs.length} devices), configurations: ${sweepCountText(sw, models, 'ss')}, rough ETA about ${fmtDur(perTarget * devs.length)}. This is a rough guess from model size and the number of passes: big models and small CPU budgets take longer; confidence values cost nothing.`;
      updateSweepSummaries();
      document.getElementById('suiteStartBtn').disabled = runs === 0;
    }

    async function openSuitePanel() {
      suiteSetupOpen = true;
      document.getElementById('suiteSection').style.display = 'block';
      document.getElementById('suiteSetup').style.display = 'block';
      document.getElementById('suiteRun').style.display = 'none';
      document.getElementById('suiteCards').innerHTML = '';
      document.getElementById('suiteMsg').textContent = '';
      buildSuiteSetup();
      document.getElementById('suiteSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
      try {  // GFLOPs of the last Estimate run make the ETA better
        const est = await (await fetch('/api/estimates')).json();
        (est.rows || []).forEach(r => { suiteGflops[r.name] = r.gflops; if (r.dynamic_input !== undefined && r.dynamic_input !== null) sweepDynamic[r.name] = r.dynamic_input; });
        updateSuiteEta();
      } catch (err) {}
    }

    function closeSuitePanel() {
      suiteSetupOpen = false;
      document.getElementById('suiteSetup').style.display = 'none';
      if (!suiteCardId && document.getElementById('suiteRun').style.display === 'none') document.getElementById('suiteSection').style.display = 'none';
    }

    function suiteStarted() {
      suiteSetupOpen = false;
      suiteCardId = null;
      lastSuiteSig = '';
      document.getElementById('suiteCards').innerHTML = '';
      document.getElementById('suiteSection').style.display = 'block';
      document.getElementById('suiteSetup').style.display = 'none';
      document.getElementById('suiteRun').style.display = 'block';
      document.getElementById('suiteProgressBox').style.display = 'block';
      document.getElementById('suiteGrid').innerHTML = '<div class="table-note">Starting...</div>';
      document.getElementById('suiteProgressText').textContent = 'Starting...';
      document.getElementById('suiteCurrent').textContent = '';
      document.getElementById('suiteProgressFill').classList.add('indeterminate');
      document.getElementById('suiteProgressFill').style.width = '';
      setRunningButtons(true);
      startPolling();
    }

    async function postSuite(url, payload) {
      const msg = document.getElementById('suiteMsg');
      try {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (data.error) {
          msg.textContent = data.error;
          document.getElementById('terminal').textContent = '[APP] ' + data.error;
          return false;
        }
        suiteStarted();
        return true;
      } catch (err) {
        msg.textContent = 'Could not start the suite: ' + err;
        return false;
      }
    }

    async function startSuite() {
      await postSuite('/api/suite', {
        env: currentEnv,
        targets: suiteChecked('suite-device'),
        models: suiteChecked('suite-model'),
        video: document.getElementById('suiteVideo').value,
        frames: parseInt(document.getElementById('suiteFrames').value) || 30,
        warmup: parseInt(document.getElementById('suiteWarmup').value) || 0,
        conf: parseFloat(document.getElementById('suiteConf').value) || 0.35,
        notes: collectNotes(suiteChecked('suite-model'), suiteChecked('suite-device')),
        sweep: readSweep('ss'),
      });
    }

    async function resumeSuite(id) {
      document.getElementById('suiteSection').style.display = 'block';
      await postSuite('/api/suite/resume', { suite_id: id });
    }

    function suiteGridHtml(st) {
      let h = '<table class="suite-grid"><tr><th></th>' + st.targets.map(t => `<th>${esc(t)}</th>`).join('') + '</tr>';
      for (const m of st.models) {
        h += `<tr><th class="m" title="${esc(m)}">${esc(shortModel(m))}</th>`;
        for (const t of st.targets) {
          const c = (st.cells[m] || {})[t] || { status: 'pending' };
          let cls = 'sg-' + c.status, txt = '&#183;', tip = c.status;
          if (c.status === 'ok') { cls = 'sg-' + (c.heat || 'green'); txt = Number(c.fps).toFixed(1); tip = 'ok: ' + txt + ' FPS'; }
          else if (c.status === 'running') {
            const ps = st.current && st.current.model === m && st.current.target === t ? st.current.pass : null;
            txt = '&#9654;' + (ps ? '<br><small>' + esc(ps.source) + ' ' + esc(ps.input) + ' ' + ps.i + '/' + ps.n + '</small>' : '');
            if (ps) tip = 'running: ' + ps.source + ' / input ' + ps.input + ' / pass ' + ps.i + ' of ' + ps.n;
          }
          else if (c.status === 'failed') txt = '&#10005;';
          else if (c.status === 'skipped') txt = '&ndash;';
          else if (c.status === 'cancelled') txt = '&oslash;';
          h += `<td class="${cls}" title="${esc(tip)}">${txt}</td>`;
        }
        h += '</tr>';
      }
      return h + '</table>';
    }

    function renderSuiteRun(st, running) {
      if (!running && suiteSetupOpen) return;  // the user is configuring the next suite: leave the panel alone
      document.getElementById('suiteSection').style.display = 'block';
      document.getElementById('suiteSetup').style.display = 'none';
      document.getElementById('suiteRun').style.display = 'block';
      const box = document.getElementById('suiteProgressBox');
      box.style.display = running ? 'block' : 'none';
      if (running) {
        const cur = st.current;
        const frac = cur && cur.total ? cur.done / cur.total : 0;
        const overall = st.total ? Math.min(1, (st.done + frac) / st.total) : 0;
        const fill = document.getElementById('suiteProgressFill');
        fill.classList.toggle('indeterminate', !st.total);
        fill.style.width = st.total ? Math.round(overall * 100) + '%' : '';
        const elapsed = st.now - st.started_ts;
        const secs = st.cell_seconds || [];
        const avg = secs.length ? secs.reduce((a, b) => a + b, 0) / secs.length : null;
        const eta = avg !== null ? fmtDur(avg * (st.total - st.done)) : 'estimating...';
        document.getElementById('suiteProgressText').textContent = st.stopping ? 'Stopping...'
          : `${st.done}/${st.total || '?'} runs - ${Math.round(overall * 100)}% - elapsed ${fmtDur(elapsed)} - ETA about ${eta}`;
        let now = 'Preparing...';
        if (st.stopping) now = 'Removing the container and rendering the partial report...';
        else if (cur && cur.model) now = `Now: ${cur.target} / ${cur.model}` + (cur.pass ? ` - ${cur.pass.source} \u00b7 ${cur.pass.input} \u00b7 pass ${cur.pass.i}/${cur.pass.n}` : '') + (cur.total ? ` - frame ${cur.done}/${cur.total}` : ' - loading model, warming up...');
        else if (st.target) now = `Starting the ${st.target} container...`;
        document.getElementById('suiteCurrent').textContent = now;
      }
      const sig = JSON.stringify(st.cells) + running + JSON.stringify(st.current && st.current.pass);
      if (sig !== lastSuiteSig) {
        lastSuiteSig = sig;
        document.getElementById('suiteGrid').innerHTML = suiteGridHtml(st);
      }
      if (!running && st.suite_id && suiteCardId !== st.suite_id) {
        suiteCardId = st.suite_id;
        loadSuites();
      }
    }

    function suiteLinks(s) {
      const b = '/api/suites/' + encodeURIComponent(s.suite_id);
      return { zip: b + '/download', html: b + '/index.html', print: b + '/print.html', csv: b + '/results.csv', json: b + '/results.json' };
    }

    function suiteCardHtml(s) {
      const l = suiteLinks(s);
      const c = s.counts || {};
      const notRun = (c.cancelled || 0) + (c.pending || 0) + (c.running || 0);
      const best = s.best ? `${shortModel(s.best.model)} on ${s.best.target}` : 'n/a';
      const badge = { completed: 'ok', stopped: 'warn', failed: 'fail', running: 'warn' }[s.status] || 'warn';
      const tiles = [[c.ok || 0, 'Succeeded', 'good'], [c.failed || 0, 'Failed', c.failed ? 'bad' : ''], [c.skipped || 0, 'Skipped', ''],
                     [notRun, 'Not run', notRun ? 'warn' : ''], [s.best ? Number(s.best.fps).toFixed(1) + ' FPS' : 'n/a', 'Fastest pair: ' + best, '']];
      const tilesHtml = tiles.map(t => `<div class="kpi-card"><div class="kpi-val ${t[2]}">${esc(t[0])}</div><div class="kpi-lbl">${esc(t[1])}</div></div>`).join('');
      return `<div class="report-card">
        <div class="rc-head">
          <div><div class="rc-title">Benchmark suite: ${s.n_models} models x ${s.n_devices} devices</div>
          <div class="rc-sub">${esc(s.suite_id)} - ${s.frames} frames per run - ${esc(s.video || 'synthetic frames')} - ${esc(fmtDur(s.elapsed_s || 0))}</div></div>
          <span class="vbadge ${badge}">${esc(String(s.status).toUpperCase())}</span>
        </div>
        <div class="rc-kpis">${tilesHtml}</div>
        <div class="rc-actions">
          ${s.has_report ? `<a class="ctrl-btn play link-btn" href="${l.html}" target="_blank" rel="noopener">Open report</a>
          <a class="ctrl-btn link-btn" href="${l.print}" target="_blank" rel="noopener">Print all (PDF)</a>
          <a class="ctrl-btn link-btn" href="${l.zip}">⬇ Download ZIP</a>
          <a class="ctrl-btn link-btn" href="${l.csv}">CSV</a>
          <a class="ctrl-btn link-btn" href="${l.json}" target="_blank" rel="noopener">JSON</a>` : ''}
          ${s.resumable ? `<button type="button" class="ctrl-btn" onclick="resumeSuite('${esc(s.suite_id)}')">↻ Resume</button>` : ''}
          ${s.has_sweep && (c.ok || 0) > 0 ? `<button type="button" class="ctrl-btn" data-id="${esc(s.suite_id)}" onclick="openRatings('suite', this.dataset.id)">&#11088; Rate detection quality</button>` : ''}
        </div>
      </div>`;
    }

    async function loadSuites() {
      try {
        const res = await fetch('/api/suites');
        suiteRows = (await res.json()).suites || [];
        const cur = suiteRows.find(s => s.suite_id === suiteCardId);
        if (cur && !suiteSetupOpen) document.getElementById('suiteCards').innerHTML = suiteCardHtml(cur);
        renderSuiteHistory();
      } catch (err) {
        document.getElementById('suiteHistory').innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(err)}</div>`;
      }
    }

    function renderSuiteHistory() {
      if (suiteRows.length === 0) {
        document.getElementById('suiteHistory').innerHTML = '<div class="table-note">No suites yet. Press "Benchmark All Scenarios" to run every model on every device.</div>';
        return;
      }
      let html = '<table class="results-table hist-table"><tr><th>Date</th><th>Suite</th><th>Models x devices</th><th>Frames</th><th>Status</th><th>ok / failed / skipped / not run</th><th>Fastest pair</th><th>Time</th><th>Actions</th></tr>';
      for (const s of suiteRows) {
        const l = suiteLinks(s);
        const c = s.counts || {};
        const notRun = (c.cancelled || 0) + (c.pending || 0) + (c.running || 0);
        const when = new Date(s.created);
        const cls = s.status === 'completed' ? 'good' : (s.status === 'failed' ? 'bad' : 'warn');
        html += `<tr><td>${esc(isNaN(when) ? s.created : when.toLocaleString())}</td><td>${esc(s.suite_id)}</td>
          <td>${s.n_models} x ${s.n_devices}</td><td>${s.frames}</td><td class="${cls}">${esc(String(s.status).toUpperCase())}</td>
          <td>${c.ok || 0} / ${c.failed || 0} / ${c.skipped || 0} / ${notRun}</td>
          <td>${s.best ? esc(shortModel(s.best.model)) + ' on ' + esc(s.best.target) + ' (' + Number(s.best.fps).toFixed(1) + ' FPS)' : 'n/a'}</td>
          <td>${esc(fmtDur(s.elapsed_s || 0))}</td>
          <td class="hist-links">${s.has_report ? `<a href="${l.html}" target="_blank" rel="noopener">Report</a><a href="${l.print}" target="_blank" rel="noopener">Print</a><a href="${l.zip}">ZIP</a><a href="${l.csv}">CSV</a><a href="${l.json}" target="_blank" rel="noopener">JSON</a>` : ''}
          ${s.resumable ? `<a href="#" onclick="resumeSuite('${esc(s.suite_id)}'); return false;">Resume</a>` : ''}
          ${s.has_sweep && (c.ok || 0) > 0 ? `<a href="#" data-id="${esc(s.suite_id)}" onclick="openRatings('suite', this.dataset.id); return false;">&#11088; Rate</a>` : ''}</td></tr>`;
      }
      document.getElementById('suiteHistory').innerHTML = html + '</table>';
    }

    // ---- Sweep settings (source resolution x model input size x confidence) ------------------------
    const SWEEP = SERVER_DATA.sweep || { defaults: {}, presets: {}, default_conf: 0.35, height_choices: [480, 720, 1080, 0], size_choices: [320, 480, 640, 800, 960] };
    const SWEEP_KEY = 'crimeDetect.sweep.v1';
    let sweepDynamic = {};   // model -> true / false, from the last Estimate run (.pt weights count as dynamic)
    const SWEEP_MODEL_SIZE = 640;

    function sweepPanelHtml(p) {
      const hs = SWEEP.height_choices.map(h => `<label><input type="checkbox" class="${p}-h" value="${h}" onchange="onSweepChange('${p}')">${h === 0 ? 'native' : h + 'p'}</label>`).join('');
      const ss = SWEEP.size_choices.map(z => `<label><input type="checkbox" class="${p}-s" value="${z}" onchange="onSweepChange('${p}')">${z}</label>`).join('');
      return `<div class="sweep-grp">Source (camera) resolution - values above the video's own are skipped</div><div class="sweep-grid">${hs}</div>
        <div class="sweep-grp">Model input size - only applied to models with dynamic input (others keep their own size)</div><div class="sweep-grid">${ss}</div>
        <div class="sweep-grp">Confidence thresholds, comma separated (the Conf field is always added)</div>
        <input type="text" id="${p}Confs" oninput="onSweepChange('${p}')">
        <div class="suite-links" style="margin: 6px 0;">Presets: <a onclick="applySweepPreset('${p}', 'quick')">Quick</a><a onclick="applySweepPreset('${p}', 'full')">Full</a><a onclick="applySweepPreset('${p}', 'defaults')">Defaults</a></div>
        <div class="checkbox-row" style="margin: 6px 0;"><input type="checkbox" id="${p}Off" onchange="onSweepChange('${p}')"><label for="${p}Off">No sweep: one configuration (old behaviour)</label></div>
        <div id="${p}Summary" class="suite-eta" style="margin: 6px 0 2px;"></div>`;
    }

    function setSweepControls(p, cfg) {
      const heights = cfg.source_heights || [], sizes = cfg.input_sizes || [];
      document.querySelectorAll('input.' + p + '-h').forEach(el => { el.checked = heights.includes(parseInt(el.value)); });
      document.querySelectorAll('input.' + p + '-s').forEach(el => { el.checked = sizes.includes(parseInt(el.value)); });
      document.getElementById(p + 'Confs').value = (cfg.conf_thresholds || []).join(', ');
      document.getElementById(p + 'Off').checked = cfg.enabled === false;
    }

    function readSweep(p) {
      if (!document.getElementById(p + 'Off')) return { enabled: false };
      if (document.getElementById(p + 'Off').checked) return { enabled: false };
      const heights = Array.from(document.querySelectorAll('input.' + p + '-h:checked')).map(el => parseInt(el.value));
      const sizes = Array.from(document.querySelectorAll('input.' + p + '-s:checked')).map(el => parseInt(el.value));
      const confs = document.getElementById(p + 'Confs').value.split(/[ ,;]+/).map(parseFloat).filter(x => !isNaN(x) && x > 0 && x < 1);
      const dflt = parseFloat(document.getElementById(p === 'bs' ? 'confInput' : 'suiteConf').value) || SWEEP.default_conf;
      if (!confs.includes(dflt)) confs.push(dflt);
      confs.sort((a, b) => a - b);
      return { enabled: true, source_heights: heights.length ? heights : [0], input_sizes: sizes, conf_thresholds: confs };
    }

    function applySweepPreset(p, name) {
      const cfg = name === 'defaults' ? SWEEP.defaults : (SWEEP.presets || {})[name];
      if (!cfg) return;
      setSweepControls(p, cfg);
      onSweepChange(p);
    }

    function onSweepChange(p) {
      try {
        const saved = JSON.parse(localStorage.getItem(SWEEP_KEY) || '{}');
        saved[p] = readSweep(p);
        localStorage.setItem(SWEEP_KEY, JSON.stringify(saved));
      } catch (e) {}
      updateSweepSummaries();
      if (p === 'ss') updateSuiteEta();
    }

    function isDynamicModel(m) {
      return sweepDynamic[m] !== undefined ? !!sweepDynamic[m] : m.toLowerCase().endsWith('.pt');
    }

    function sweepSizes(sw, dyn) {
      return dyn ? Array.from(new Set((sw.input_sizes || []).concat([SWEEP_MODEL_SIZE]))) : [SWEEP_MODEL_SIZE];
    }

    function sweepCount(sw, dyn) {
      if (!sw || sw.enabled === false) return 1;
      return sw.source_heights.length * sweepSizes(sw, dyn).length * sw.conf_thresholds.length;
    }

    function sweepCountText(sw, models, p) {
      if (!sw || sw.enabled === false) return 'sweep off: one configuration per pair';
      const dyn = models.filter(isDynamicModel).length, fixed = models.length - dyn;
      const parts = [];
      if (fixed) parts.push(`${sweepCount(sw, false)} for a fixed-input model`);
      if (dyn) parts.push(`${sweepCount(sw, true)} for a dynamic-input model`);
      return `${parts.join(', ')} (${sw.source_heights.length} resolutions x input sizes x ${sw.conf_thresholds.length} confidences)`;
    }

    function cellSeconds(m, frames, warmup, sw) {
      const E = SUITE_EST;
      const g = suiteGflops[m] || E.default_gflops;
      const mb = (SERVER_DATA.model_sizes || {})[m] || 0;
      const torch = m.toLowerCase().endsWith('.pt') ? E.torch_start_s : 0;
      if (!sw || sw.enabled === false) {
        return torch + E.cell_overhead_s + E.load_s_per_mb * mb + (frames + warmup) * (E.host_ms_base + E.host_ms_per_gflop * g) / 1000;
      }
      const sizes = sweepSizes(sw, isDynamicModel(m));
      const area = sizes.reduce((a, z) => a + Math.pow(z / SWEEP_MODEL_SIZE, 2), 0) / sizes.length;
      const host = E.host_ms_base + E.host_ms_per_gflop * g * area;
      const passes = sw.source_heights.length * sizes.length;
      return torch + E.cell_overhead_s + E.load_s_per_mb * mb + warmup * host / 1000
        + passes * (frames * (host + E.decode_ms) / 1000 + E.pass_overhead_s + E.extra_warmup * host / 1000);
    }

    function updateSweepSummaries() {
      const el = document.getElementById('bsSummary');
      if (!el || !document.getElementById('modelSelect')) return;
      const models = selectedBenchModels();
      const sw = readSweep('bs');
      const frames = parseInt(document.getElementById('framesInput').value) || 50;
      let secs = SUITE_EST.container_start_s;
      for (const m of models) secs += cellSeconds(m, frames, 3, sw);
      el.textContent = models.length
        ? `Configurations: ${sweepCountText(sw, models, 'bs')}; ${frames} frames per pass, rough ETA about ${fmtDur(secs)} for ${models.length} model(s).`
        : 'Select a model.';
      const el2 = document.getElementById('ssSummary');
      if (el2) {
        const ms = suiteChecked('suite-model').filter(isRunnable);
        el2.textContent = ms.length ? 'Configurations: ' + sweepCountText(readSweep('ss'), ms, 'ss') + ' per device.' : '';
      }
    }

    function buildSweepPanels() {
      let saved = {};
      try { saved = JSON.parse(localStorage.getItem(SWEEP_KEY) || '{}') || {}; } catch (e) { saved = {}; }
      for (const p of ['bs', 'ss']) {
        document.getElementById(p + 'Sweep').innerHTML = sweepPanelHtml(p);
        setSweepControls(p, saved[p] && saved[p].source_heights ? saved[p] : SWEEP.defaults);
      }
      updateSweepSummaries();
    }

    // ---- Manual quality ratings ("Rate detection quality") --------------------------------------
    let ratingsCtx = null;
    let ratingsMsg = '';

    function closeLightbox() { document.getElementById('lightbox').style.display = 'none'; }
    function openLightbox(src) {
      document.getElementById('lightboxImg').src = src;
      document.getElementById('lightbox').style.display = 'flex';
    }
    function closeRatings() {
      document.getElementById('ratingsModal').style.display = 'none';
      document.body.style.overflow = '';
      ratingsCtx = null;
    }
    document.addEventListener('keydown', ev => {
      if (ev.key !== 'Escape') return;
      if (document.getElementById('lightbox').style.display === 'flex') closeLightbox();
      else if (document.getElementById('ratingsModal').style.display === 'block') closeRatings();
    });

    async function openRatings(kind, id) {
      ratingsMsg = '';
      document.getElementById('ratingsModal').style.display = 'block';
      document.body.style.overflow = 'hidden';
      document.getElementById('ratingsBody').innerHTML = '<div class="table-note">Loading the configurations...</div>';
      await loadRatingsView(kind, id);
    }

    async function loadRatingsView(kind, id) {
      const body = document.getElementById('ratingsBody');
      try {
        const res = await fetch('/api/ratings/view?' + kind + '=' + encodeURIComponent(id));
        const data = await res.json();
        if (data.error) { body.innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(data.error)}</div>`; return; }
        ratingsCtx = { kind: kind, id: id, data: data };
        renderRatings();
      } catch (err) {
        body.innerHTML = `<div class="table-note" style="color: var(--danger)">${esc(err)}</div>`;
      }
    }

    function fmtNum(v, nd) { return v === null || v === undefined ? 'n/a' : Number(v).toFixed(nd); }

    function ratingItemHtml(it, ii) {
      const frames = (it.sample_frames || []).join(', ');
      const devs = (it.devices || []).map(d => `${esc(d.device)}: <b>${esc(d.best)}</b> <span style="color: var(--text-muted)">(${esc(RULE_NAMES[d.rule] || d.rule)})</span>`).join('<br>');
      const rows = it.configs.map((c, ci) => {
        const thumbs = (c.samples || []).map(s => `<img loading="lazy" src="${esc(it.base_url + s)}" alt="${esc(c.short)}" onclick="openLightbox(this.src)">`).join('');
        const dev = Object.entries(c.per_device || {}).filter(([, v]) => v).map(([t, v]) => `${esc(t)} ${v.fps} FPS${v.realtime ? '' : ' (slow)'}${v.best ? ' &#9733;' : ''}`).join('<br>');
        const isBest = Object.values(c.per_device || {}).some(v => v && v.best);
        return `<tr class="${isBest ? 'rate-best' : ''}"><td class="rate-cfg"><b>${esc(c.short)}</b><br>${esc(c.source_text)}, input ${esc(c.input)}, conf ${esc(c.conf)}
            <br><span style="color: var(--text-muted)">stability ${fmtNum(c.stability, 2)} &middot; agreement ${fmtNum(c.agreement_f1, 2)} &middot; temporal ${fmtNum(c.temporal, 2)} &middot; ${esc(c.dets_per_frame)} boxes/frame</span></td>
          <td class="rate-thumbs">${thumbs}</td>
          <td style="text-align: left; white-space: normal; font-family: inherit;">${dev}</td>
          <td><input type="number" class="rate-in" min="1" max="5" step="1" placeholder="-" data-f="coverage" data-i="${ii}" data-c="${ci}" value="${c.coverage === null || c.coverage === undefined ? '' : esc(c.coverage)}"></td>
          <td><input type="number" class="rate-in" min="0" step="1" placeholder="-" data-f="duplicates" data-i="${ii}" data-c="${ci}" value="${c.duplicates === null || c.duplicates === undefined ? '' : esc(c.duplicates)}"></td></tr>`;
      }).join('');
      return `<div class="rate-item"><h3>${esc(it.model)}</h3>
        <div class="table-note" style="margin: 0 0 4px;">Video ${esc(it.video)}. Sample frames ${esc(frames)}: the same frames in every configuration (the ones with the most detections in the reference configuration). Click a picture to enlarge.${it.locked ? ' The input size of this model is locked, so only resolution and confidence differ.' : ''}</div>
        <div class="table-note" style="margin: 0 0 8px;">Best configuration now:<br>${devs}</div>
        <div class="table-wrap" style="overflow-x: auto;"><table class="results-table rate-table"><tr><th>Configuration</th><th class="note-th">Sample frames</th><th class="note-th">Speed per device</th><th title="${esc(ratingsCtx.data.help.coverage)}">Coverage (1-5)</th><th title="${esc(ratingsCtx.data.help.duplicates)}">Duplicates</th></tr>${rows}</table></div>
        <div class="btn-row" style="margin-top: 6px;"><button type="button" class="ctrl-btn" data-i="${ii}" onclick="clearRatingInputs(this.dataset.i)">Clear the ratings of this model</button></div></div>`;
    }

    function renderRatings() {
      const d = ratingsCtx.data;
      const items = d.items.map((it, ii) => ratingItemHtml(it, ii)).join('');
      document.getElementById('ratingsBody').innerHTML = `
        <div class="rate-help"><b>Coverage (1-5):</b> ${esc(d.help.coverage)}.<br><b>Duplicates:</b> ${esc(d.help.duplicates)}.<br>
          Leave a field blank if you did not rate it. Ratings belong to the video, the model and the configuration, so they apply to every report and suite of that video and model, on any device.
          If a model has ratings, only its rated configurations compete for the best configuration (highest coverage, then fewest duplicates, then real-time, then stability); without ratings the automatic real-time + stable rule decides.</div>
        ${items}
        <div id="rateMsg" class="rate-msg">${ratingsMsg}</div>
        <div class="btn-row"><button type="button" class="action-btn btn-primary" style="flex: none; padding: 9px 22px;" onclick="saveRatings()">Save ratings and update the reports</button>
        <button type="button" class="action-btn btn-secondary" style="flex: none; padding: 9px 22px;" onclick="closeRatings()">Close</button></div>`;
    }

    function clearRatingInputs(ii) {
      document.querySelectorAll('input.rate-in[data-i="' + ii + '"]').forEach(el => { el.value = ''; });
    }

    async function saveRatings() {
      const msg = document.getElementById('rateMsg');
      const entries = [];
      const inputs = Array.from(document.querySelectorAll('input.rate-in'));
      for (const el of inputs) {
        const it = ratingsCtx.data.items[parseInt(el.dataset.i)], c = it.configs[parseInt(el.dataset.c)];
        let e = entries.find(x => x._i === el.dataset.i && x._c === el.dataset.c);
        if (!e) { e = { _i: el.dataset.i, _c: el.dataset.c, video: it.video, model: it.model, source: c.source, input: c.input, conf: c.conf, coverage: '', duplicates: '' }; entries.push(e); }
        const v = el.value.trim();
        if (v !== '') {
          const n = Number(v);
          const bad = !Number.isInteger(n) || (el.dataset.f === 'coverage' ? (n < 1 || n > 5) : n < 0);
          if (bad) { msg.innerHTML = '<span style="color: var(--danger)">' + esc((c.short) + ': ' + (el.dataset.f === 'coverage' ? 'coverage must be a whole number from 1 to 5' : 'duplicates must be a whole number, 0 or more')) + '</span>'; el.focus(); return; }
        }
        e[el.dataset.f] = v === '' ? '' : Number(v);
      }
      entries.forEach(e => { delete e._i; delete e._c; });
      msg.textContent = 'Saving and re-rendering the reports...';
      const payload = { ratings: entries };
      payload[ratingsCtx.kind] = ratingsCtx.id;
      try {
        const res = await fetch('/api/ratings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (data.error) { msg.innerHTML = '<span style="color: var(--danger)">' + esc(data.error) + '</span>'; return; }
        const r = data.rerendered || {};
        ratingsMsg = '<span style="color: var(--accent)">Saved (' + data.changed + ' rating(s) changed). Re-rendered ' + (r.reports || []).length + ' report(s) and ' + (r.suites || []).length + ' suite(s)'
          + ((r.skipped || []).length ? '; the running suite ' + esc(r.skipped.join(', ')) + ' picks them up when it finishes' : '') + '. The best configuration per device is shown above.</span>';
        await loadRatingsView(ratingsCtx.kind, ratingsCtx.id);
        refreshAfterRatings();
      } catch (err) {
        msg.innerHTML = '<span style="color: var(--danger)">' + esc(err) + '</span>';
      }
    }

    function refreshAfterRatings() {
      Array.from(renderedReports).forEach(id => refreshReportCard(id));
      loadReports();
      loadSuites();
    }

    // Instantly populate UI from pre-rendered data
    window.addEventListener('DOMContentLoaded', () => {
      populateUI(SERVER_DATA);
      buildSweepPanels();
      fetch('/api/estimates').then(r => r.json()).then(est => {
        (est.rows || []).forEach(r => { suiteGflops[r.name] = r.gflops; if (r.dynamic_input !== undefined && r.dynamic_input !== null) sweepDynamic[r.name] = r.dynamic_input; });
        updateSweepSummaries();
      }).catch(() => {});
      setMode('inference');
      loadReports();
      loadSuites();
      fetchLogs();  // re-attach to a running job / show results of the last one after a page reload
    });
  </script>
</body>
</html>
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model Inference Benchmark Live Simulator App")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind (default 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    port = args.port
    threading.Thread(target=check_docker_status_cached, daemon=True).start()

    for p in range(port, port + 20):
        try:
            server = ThreadedHTTPServer(("", p), AppHandler)
            port = p
            break
        except OSError:
            continue

    url = f"http://localhost:{port}"
    print("=" * 70)
    print(f"  MODEL INFERENCE BENCHMARK LIVE SIMULATOR & PLAYER LAUNCHED")
    print(f"  URL: {url}")
    print("  Press Ctrl+C to stop the server")
    print("=" * 70)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    main()
