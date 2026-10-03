"""
Benchmark suite: every selected model on every selected target, written as a multipage benchmark report
(results/reports/suite_<timestamp>/index.html, one page per device and per model, methodology, print.html,
results.csv / results.json / suite.json, and every per-run report under runs/).

    python scripts/run_benchmark_matrix.py --targets all --models all --frames 30 --video videos/input/sample.mp4
    python scripts/run_benchmark_matrix.py --targets arm64-cpu rk3588-npu --models models/a.onnx models/b.onnx --local
    python scripts/run_benchmark_matrix.py --resume suite_20260930_120000        # only the cells that are not done yet
    python scripts/run_benchmark_matrix.py --render-only suite_20260930_120000   # rebuild the site from suite.json
    python scripts/run_benchmark_matrix.py --targets jetson --models models/a.onnx --sweep-preset quick   # fewer configurations
    python scripts/run_benchmark_matrix.py --targets jetson --models models/a.onnx --no-sweep            # one configuration per pair
    python scripts/run_benchmark_matrix.py --targets jetson --models models/a.onnx --notes-file my_notes.json

Notes per model/device pair (optional): --notes-file is a UTF-8 JSON {"a.onnx": {"jetson": "free text"}}. They are stored in
suite.json (every cell has "notes") and appear as a Notes column in the report. Edit a cell's "notes" in suite.json and run
--render-only to update the report; --resume keeps them.

Sweeps: every model/device pair is tested over source resolutions x model input sizes (only models with a dynamic input) x
confidence thresholds (defaults: benchmark.sweep in configs/detection.yaml; --source-heights, --input-sizes, --conf-thresholds,
--sweep-preset quick|full, --no-sweep). The overview shows the best configuration per pair with the reason it was picked; user
ratings (scripts/rate_configs.py or the web app) take priority over the automatic real-time + stable rule. --render-only re-renders
the suite AND its per-run reports from the stored data, so new ratings show up everywhere.

Each target runs in its own Docker service (CPU/RAM limits of the device) unless --local is given; models run one after
another, never in parallel. Ctrl+C / SIGTERM stops the suite, marks the rest as cancelled and still renders the report.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import sweep as sweep_mod
from src.benchmark.notes import load_notes_file, note_for
from src.benchmark.suite import (SUITE_ID_RE, SuiteRunner, load_manifest, mark_stopped, new_manifest,
                                 new_suite_id, pick_models, prune_orphan_runs, render_suite, save_manifest, summarize)
from src.config import TargetConfig, load_yaml
from src.utils.docker_runner import is_docker_daemon_running
from src.utils.logger import get_logger

logger = get_logger("BenchmarkSuite")
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark models x targets and write a multipage benchmark report")
    parser.add_argument("--targets", nargs="+", default=["all"], help="Target keys from targets.yaml, or 'all'")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model paths / names in models/, or 'all'")
    parser.add_argument("--frames", type=int, default=30, help="Frames per run")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per run")
    parser.add_argument("--video", type=str, default=None, help="Video path (default: synthetic frames)")
    parser.add_argument("--conf", type=float, default=None, help="Override the confidence threshold")
    parser.add_argument("--no-sweep", action="store_true", help="One configuration per pair (the config file's input size, native resolution)")
    parser.add_argument("--sweep-preset", choices=sorted(sweep_mod.PRESETS), default=None,
                        help="quick: native + 720p, model default input size; full: 480p-native x 320-960 input sizes")
    parser.add_argument("--source-heights", nargs="+", default=None, metavar="H", help="Source resolutions as frame heights, e.g. 720 1080 native")
    parser.add_argument("--input-sizes", nargs="+", default=None, metavar="S",
                        help="Model input sizes (multiples of 32) for models with dynamic input; 'default' = the model's own size only")
    parser.add_argument("--conf-thresholds", nargs="+", default=None, metavar="C", help="Confidence thresholds (the --conf / config value is added)")
    parser.add_argument("--local", action="store_true", help="Run on this machine instead of in the target's Docker service")
    parser.add_argument("--resume", metavar="SUITE_ID", help="Continue a stopped suite: reruns only cells that are not ok/skipped")
    parser.add_argument("--render-only", metavar="SUITE_ID", help="Rebuild the report site of an existing suite from its suite.json")
    parser.add_argument("--notes-file", metavar="PATH", help="UTF-8 JSON {model file: {target: note}}: notes per model/device pair "
                        "(stored in suite.json; for a new suite only)")
    parser.add_argument("--mark-stopped", action="store_true",
                        help="With --render-only: mark pending/running cells as cancelled first (used after killing a suite)")
    return parser.parse_args()


def suite_folder(suite_id: str) -> Path:
    if not SUITE_ID_RE.match(suite_id) or not (REPORTS_DIR / suite_id / "suite.json").is_file():
        logger.error(f"No suite '{suite_id}' in {REPORTS_DIR}")
        sys.exit(2)
    return REPORTS_DIR / suite_id


def render_only(suite_id: str, mark: bool) -> None:
    folder = suite_folder(suite_id)
    manifest = load_manifest(folder)
    if mark and manifest["status"] == "running":
        mark_stopped(manifest, "cancelled: the suite was stopped")
    if mark:
        prune_orphan_runs(folder, manifest)
    save_manifest(folder, manifest)
    render_suite(folder, manifest)
    print(f"SUITE_DONE {suite_id}", flush=True)
    logger.info(f"Rendered {folder / 'index.html'} (status {manifest['status']})")


def install_stop_handlers() -> None:
    """SIGTERM behaves like Ctrl+C so the runner can remove its container and render the partial report."""
    def handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), handler)
            except (ValueError, OSError):
                pass


def main():
    for stream in (sys.stdout, sys.stderr):  # container output is UTF-8; a Windows console / pipe may default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    if args.render_only:
        render_only(args.render_only, args.mark_stopped)
        return

    target_cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")
    if args.resume:
        folder = suite_folder(args.resume)
        manifest = load_manifest(folder)
        local = manifest["config"].get("execution") == "local"
    else:
        targets = target_cfg.available_targets if args.targets == ["all"] else args.targets
        unknown = [t for t in targets if t not in target_cfg.targets]
        if unknown:
            logger.error(f"Unknown target(s) {unknown}. Available: {target_cfg.available_targets}")
            sys.exit(2)
        models, skipped = pick_models(args.models, PROJECT_ROOT / "models")
        if not models:
            logger.error("No runnable models (.onnx/.pt) selected")
            sys.exit(2)
        for s in skipped:
            logger.warning(f"Skipping {s['model']}: {s['reason']}")
        local = args.local
        notes = {}
        if args.notes_file:
            try:
                raw_notes = load_notes_file(args.notes_file)
            except (OSError, ValueError) as e:
                logger.error(f"Cannot read --notes-file '{args.notes_file}': {e}")
                sys.exit(2)
            known = {Path(m).name for m in models + [s["model"] for s in skipped]}
            for m in raw_notes:
                if m not in known:
                    logger.warning(f"--notes-file: no model '{m}' in this suite, its notes are ignored")
            notes = {m: {t: n for t in targets if (n := note_for(raw_notes, m, t))} for m in sorted(known)}
            notes = {m: per for m, per in notes.items() if per}
        det_cfg = load_yaml(PROJECT_ROOT / "configs" / "detection.yaml")
        default_conf = args.conf if args.conf is not None else float((det_cfg.get("model") or {}).get("conf_threshold", 0.35))
        raw_sweep = dict(sweep_mod.PRESETS[args.sweep_preset]) if args.sweep_preset else {}
        for key, val in (("source_heights", args.source_heights), ("input_sizes", args.input_sizes), ("conf_thresholds", args.conf_thresholds)):
            if val is not None:
                raw_sweep[key] = [x for x in val if str(x).lower() not in ("default", "none")] if key == "input_sizes" else val
        try:
            sweep = None if args.no_sweep else sweep_mod.normalize_sweep(raw_sweep, default_conf, (det_cfg.get("benchmark") or {}).get("sweep"))
        except ValueError as e:
            logger.error(f"Invalid sweep settings: {e}")
            sys.exit(2)
        config = {"targets": targets, "models": models, "frames": args.frames, "warmup": args.warmup,
                  "video": args.video, "conf": args.conf, "required_fps": None, "notes": notes,
                  "execution": "local" if local else "docker", "sweep": sweep}
        suite_id = new_suite_id(REPORTS_DIR)
        folder = REPORTS_DIR / suite_id
        (folder / "runs").mkdir(parents=True)
        manifest = new_manifest(suite_id, config, skipped, target_cfg, PROJECT_ROOT / "models")
        save_manifest(folder, manifest)
    if not local and not is_docker_daemon_running():
        logger.error("Docker daemon is not running. Start Docker Desktop or use --local.")
        sys.exit(1)

    install_stop_handlers()
    runner = SuiteRunner(PROJECT_ROOT, folder, manifest, local=local)
    started = time.time()
    try:
        status = runner.run()
    except KeyboardInterrupt:
        logger.warning(f"Suite stopped. Resume with: python scripts/run_benchmark_matrix.py --resume {manifest['suite_id']}")
        sys.exit(130)
    s = summarize(manifest)
    logger.info(f"Suite {status}: {s['counts']['ok']} ok, {s['counts']['failed']} failed, {s['counts']['skipped']} skipped "
                f"in {time.time() - started:.0f}s. Open {folder / 'index.html'}")
    if s["counts"]["ok"] == 0 and s["counts"]["skipped"] != s["total"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
