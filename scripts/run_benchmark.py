import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil
from src.config import TargetConfig, load_yaml
from src.runtimes import get_detector
from src.video import VideoReader
from src.benchmark import (
    BenchmarkProfiler, save_benchmark_report, save_comparison_report,
    print_summary_table, print_comparison_table, print_sweep_table, build_report, write_report_bundle, render_samples,
)
from src.benchmark import sweep as sweep_mod
from src.benchmark.notes import clean_note, load_notes_file, note_for
from src.benchmark.sweep_runner import build_sweep_report, detection_view, run_passes
from src.simulation import SimulatedDetector
from src.utils.resource_limiter import setup_resource_limiter
from src.utils.docker_runner import build_docker_run_command, is_docker_daemon_running
from src.utils.logger import get_logger

logger = get_logger("BenchmarkHarness")

PROGRESS_EVERY = 5  # frames between "PROGRESS done/total" lines (parsed by the web app)

def parse_args():
    parser = argparse.ArgumentParser(description="Edge Detection Model Benchmark Harness")
    parser.add_argument("--target", type=str, default="x86-cpu", help="Target hardware key from targets.yaml")
    parser.add_argument("--list-targets", action="store_true", help="List all available target platforms")
    parser.add_argument("--docker", action="store_true", help="Execute simulation inside Docker container")
    parser.add_argument("--model", type=str, nargs="+", default=["models/best-yolo11-seg.pt"],
                        help="Model weights path(s). Several models are benchmarked one after another and compared")
    parser.add_argument("--video", type=str, default=None, help="Video path for benchmark (or synthetic frames)")
    parser.add_argument("--config", type=str, default="configs/detection.yaml", help="Detection config file")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override the confidence threshold from the config (it is always one of the swept thresholds)")
    # Sweep of source resolution x model input size x confidence (defaults: benchmark.sweep in the config file)
    parser.add_argument("--no-sweep", action="store_true",
                        help="Test one configuration only (the config file's input size, the video's own resolution, --conf)")
    parser.add_argument("--sweep-preset", choices=sorted(sweep_mod.PRESETS), default=None,
                        help="quick: native + 720p, model default input size; full: 480p-native x 320-960 input sizes")
    parser.add_argument("--source-heights", nargs="+", default=None, metavar="H",
                        help="Source (camera) resolutions as frame heights, e.g. 720 1080 native")
    parser.add_argument("--input-sizes", nargs="+", default=None, metavar="S",
                        help="Model input sizes (multiples of 32), only for models with dynamic input; 'default' = the model's own size only")
    parser.add_argument("--conf-thresholds", nargs="+", default=None, metavar="C",
                        help="Confidence thresholds to test (no extra inference); --conf / the config value is always added")
    parser.add_argument("--max-ram-mb", type=int, default=None, help="RAM memory limit in MB (e.g. 2048 for 2GB)")
    parser.add_argument("--max-vram-mb", type=int, default=None, help="GPU VRAM memory limit in MB (e.g. 1024 for 1GB)")
    parser.add_argument("--no-simulate", action="store_true", help="Report host timings only (skip device latency estimation)")
    parser.add_argument("--no-report", action="store_true", help="Skip the report bundle (results/reports/<id>/)")
    parser.add_argument("--summary-json", type=str, default=None,
                        help="Write all runs of this invocation to this JSON file (relative to project root)")
    # Used by the benchmark suite runner (scripts/run_benchmark_matrix.py)
    parser.add_argument("--report-root", type=str, default="results/reports",
                        help="Folder that receives the report bundles (relative to project root)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Sample frames kept per report (default 3; 0 keeps none)")
    parser.add_argument("--link-samples", action="store_true",
                        help="report.html references samples/*.jpg instead of embedding them as base64")
    parser.add_argument("--back-link", type=str, default=None,
                        help="Relative href of a '<- back to suite' link shown at the top of report.html")
    parser.add_argument("--no-metrics-files", action="store_true",
                        help="Do not write the legacy results/metrics/benchmark_*.json/csv files")
    # Free-text notes per model/device pair, shown as a Notes column in the reports
    parser.add_argument("--notes", type=str, default=None,
                        help="Note for the benchmarked model on --target (single --model only; use --notes-file for several)")
    parser.add_argument("--notes-file", type=str, default=None,
                        help="UTF-8 JSON {model file: {target: note}} with the notes of the models/targets of this run")
    args = parser.parse_args()
    if args.notes is not None and len(args.model) != 1:
        parser.error("--notes needs exactly one --model; use --notes-file for several models")
    try:
        sweep_from_args(args, args.conf if args.conf is not None else 0.35, {})
    except ValueError as e:
        parser.error(str(e))
    return args

def sweep_from_args(args, default_conf: float, base_cfg: dict):
    """Normalised sweep settings from the flags and the config file defaults, or None for --no-sweep."""
    if args.no_sweep:
        return None
    raw = dict(sweep_mod.PRESETS[args.sweep_preset]) if args.sweep_preset else {}
    if args.source_heights is not None:
        raw["source_heights"] = args.source_heights
    if args.input_sizes is not None:
        raw["input_sizes"] = [x for x in args.input_sizes if str(x).lower() not in ("default", "none")]
    if args.conf_thresholds is not None:
        raw["conf_thresholds"] = args.conf_thresholds
    return sweep_mod.normalize_sweep(raw, default_conf, base_cfg)


def forwarded_args(args, models) -> list:
    sub_args = ["--target", args.target, "--frames", str(args.frames), "--warmup", str(args.warmup),
                "--config", args.config, "--model", *models]
    if args.video:
        sub_args.extend(["--video", args.video])
    if args.conf is not None:
        sub_args.extend(["--conf", str(args.conf)])
    if args.max_ram_mb:
        sub_args.extend(["--max-ram-mb", str(args.max_ram_mb)])
    if args.max_vram_mb:
        sub_args.extend(["--max-vram-mb", str(args.max_vram_mb)])
    if args.no_simulate:
        sub_args.append("--no-simulate")
    if args.no_report:
        sub_args.append("--no-report")
    sub_args.extend(["--report-root", args.report_root])
    if args.max_samples is not None:
        sub_args.extend(["--max-samples", str(args.max_samples)])
    if args.link_samples:
        sub_args.append("--link-samples")
    if args.back_link:
        sub_args.extend(["--back-link", args.back_link])
    if args.no_metrics_files:
        sub_args.append("--no-metrics-files")
    if args.notes_file:  # the notes travel in a file, never as command-line text
        sub_args.extend(["--notes-file", args.notes_file])
    if args.no_sweep:
        sub_args.append("--no-sweep")
    if args.sweep_preset:
        sub_args.extend(["--sweep-preset", args.sweep_preset])
    if args.source_heights is not None:
        sub_args.extend(["--source-heights", *args.source_heights])
    if args.input_sizes is not None:
        sub_args.extend(["--input-sizes", *args.input_sizes])
    if args.conf_thresholds is not None:
        sub_args.extend(["--conf-thresholds", *args.conf_thresholds])
    return sub_args

def resolve_note(args, model_arg: str) -> str:
    """Note of (model, --target): --notes, else the --notes-file entry; "" when there is none."""
    if args.notes is not None:
        return clean_note(args.notes)
    if args.notes_file:
        try:
            return note_for(load_notes_file(resolve(args.notes_file)), model_arg, args.target)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read the notes file '{args.notes_file}': {e}")
    return ""

def write_temp_notes(args) -> Path:
    """results/reports/_notes/notes_<timestamp>.json with the --notes text (bind-mounted, so the container can read it)."""
    folder = PROJECT_ROOT / "results" / "reports" / "_notes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"notes_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps({Path(args.model[0]).name: {args.target: clean_note(args.notes)}}, ensure_ascii=False),
                    encoding="utf-8")
    return path

def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p

def run_single(args, target_cfg: TargetConfig, model_arg: str) -> dict:
    target_info = target_cfg.get_target(args.target)
    device = None if args.no_simulate else target_cfg.device_profile(args.target)
    logger.info(f"Target selected: '{args.target}' ({target_info.get('description', '')})")

    # Load configurations
    det_cfg = load_yaml(PROJECT_ROOT / args.config)
    if args.conf is not None:
        det_cfg.setdefault("model", {})["conf_threshold"] = args.conf
    res_cfg = det_cfg.get("resources", {})

    # Determine RAM and VRAM limits (CLI > Config > targets.yaml)
    max_ram_mb = args.max_ram_mb or res_cfg.get("max_ram_mb") or target_info.get("ram_limit_mb")
    max_vram_mb = args.max_vram_mb or res_cfg.get("max_vram_mb") or target_info.get("vram_limit_mb")

    # Initialize Resource Limiter & Memory Tracker
    memory_tracker = setup_resource_limiter(max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb)
    process = psutil.Process()

    # Validate inputs before the (slow) model load
    model_path = resolve(model_arg)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    video_path = resolve(args.video) if args.video else None
    if video_path and not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load Model Detector (cold start)
    rss_before_load = process.memory_info().rss / 1024 ** 2
    t_load = time.perf_counter()
    detector = get_detector(args.target, str(model_path), det_cfg, device=device)
    model_load_ms = getattr(detector, "load_ms", None) or (time.perf_counter() - t_load) * 1000.0
    rss_after_load = process.memory_info().rss / 1024 ** 2

    # Model Warmup (the first call is the cold-start inference)
    logger.info(f"Warming up model with {args.warmup} iterations...")
    inner = getattr(detector, "inner", detector)
    input_size = tuple(getattr(inner, "input_size", None) or det_cfg.get("model", {}).get("input_size", [640, 640]))
    dummy = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    first_inference_ms = None
    for i in range(args.warmup):
        t0 = time.perf_counter()
        detector.predict(dummy)
        if i == 0:
            first_inference_ms = (time.perf_counter() - t0) * 1000.0
    rss_after_warmup = process.memory_info().rss / 1024 ** 2

    default_conf = float(det_cfg.get("model", {}).get("conf_threshold", 0.35))
    sweep = sweep_from_args(args, default_conf, (det_cfg.get("benchmark") or {}).get("sweep") or {})
    if sweep is not None:
        cold = {
            "model_load_ms": round(model_load_ms, 1),
            "first_inference_ms": round(first_inference_ms, 1) if first_inference_ms is not None else None,
            "rss_before_load_mb": round(rss_before_load, 1),
            "rss_after_load_mb": round(rss_after_load, 1),
            "rss_after_warmup_mb": round(rss_after_warmup, 1),
        }
        return run_sweep_benchmark(args, target_cfg, target_info, device, det_cfg, detector, model_path, video_path, sweep,
                                   default_conf, cold, max_ram_mb, max_vram_mb, model_arg)

    # Frame source: frames are streamed one by one so RAM reflects the model, not a preloaded video
    reader = VideoReader(video_path) if video_path else None
    if reader:
        source = {"type": "video", "file": video_path.name, "width": reader.width, "height": reader.height,
                  "fps": round(reader.fps, 2), "total_frames": reader.total_frames}
        total = args.frames if reader.total_frames <= 0 else min(args.frames, reader.total_frames)
    else:
        logger.info(f"No video provided. Generating {args.frames} synthetic test frames ({input_size[1]}x{input_size[0]})...")
        source = {"type": "synthetic", "file": None, "width": input_size[1], "height": input_size[0],
                  "fps": None, "total_frames": args.frames}
        total = args.frames

    # Run Benchmark Profiling
    simulated = isinstance(detector, SimulatedDetector)
    profiler = BenchmarkProfiler(
        target_name=args.target,
        model_name=model_path.name,
        memory_tracker=memory_tracker,
        device=device,
        model_profile=detector.model_profile if simulated else None,
        **({"sample_count": args.max_samples} if args.max_samples is not None else {}),
    )
    logger.info(f"Running benchmark on {total} frames on target '{args.target}'...")

    profiler.start()
    try:
        for idx in range(total):
            decode_ms = None
            if reader:
                t_dec = time.perf_counter()
                ok, frame, _ = reader.read_frame()
                if not ok or frame is None:
                    break
                decode_ms = (time.perf_counter() - t_dec) * 1000.0
            else:
                frame = np.random.randint(0, 256, (input_size[0], input_size[1], 3), dtype=np.uint8)
            result = detector.predict(frame)
            profiler.record_frame(result, frame_index=idx, decode_ms=decode_ms, frame=frame)
            if (idx + 1) % PROGRESS_EVERY == 0 or idx + 1 == total:
                print(f"PROGRESS {idx + 1}/{total}", flush=True)
    finally:
        if reader:
            reader.release()

    metrics = profiler.finish()
    note = resolve_note(args, model_arg)
    metrics["notes"] = note
    if "error" in metrics:
        raise RuntimeError(f"{metrics['error']} (the video ended before the first frame)")
    metrics["cold_start"] = {
        "model_load_ms": round(model_load_ms, 1),
        "first_inference_ms": round(first_inference_ms, 1) if first_inference_ms is not None else None,
        "rss_before_load_mb": round(rss_before_load, 1),
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_after_warmup_mb": round(rss_after_warmup, 1),
    }

    # Output & Export Results
    print_summary_table(metrics)
    if not args.no_report:
        report = build_report(
            metrics=metrics, frame_records=profiler.frame_records, target_name=args.target,
            target_info=target_info, device=target_cfg.device_profile(args.target), detector=detector,
            model_path=model_path, det_cfg=det_cfg, frames_requested=args.frames, warmup=args.warmup,
            source=source, cold_start=metrics["cold_start"], notes=note,
        )
        samples = render_samples(profiler.top_frames, report)
        write_report_bundle(report, profiler.frame_records, samples, resolve(args.report_root),
                            link_samples=args.link_samples,
                            back_link=(args.back_link, "back to suite") if args.back_link else None)
        metrics["report_id"] = report["report_id"]
        metrics["report_overall"] = report["verdict"]["overall"]
        logger.info(f"REPORT {report['report_id']}")
    if not args.no_metrics_files:
        json_path, csv_path = save_benchmark_report(metrics, PROJECT_ROOT / "results/metrics")
        logger.info(f"Saved JSON metrics to: {json_path}")
        logger.info(f"Saved CSV metrics to:  {csv_path}")
    return metrics

def run_sweep_benchmark(args, target_cfg, target_info, device, det_cfg, detector, model_path, video_path, sweep, default_conf,
                        cold_start, max_ram_mb, max_vram_mb, model_arg) -> dict:
    """Sweep path of run_single: source resolution x model input size passes, confidence derived by filtering."""
    if video_path:
        reader = VideoReader(video_path)
        source = {"type": "video", "file": video_path.name, "width": reader.width, "height": reader.height,
                  "fps": round(reader.fps, 2), "total_frames": reader.total_frames}
        total = args.frames if reader.total_frames <= 0 else min(args.frames, reader.total_frames)
        reader.release()
    else:
        size = tuple(getattr(getattr(detector, "inner", detector), "input_size", None) or (640, 640))
        logger.info(f"No video provided. Generating {args.frames} synthetic test frames ({size[1]}x{size[0]})...")
        source = {"type": "synthetic", "file": None, "width": size[1], "height": size[0], "fps": None, "total_frames": args.frames}
        total = args.frames
    simulated = isinstance(detector, SimulatedDetector)
    samples_n = args.max_samples if args.max_samples is not None else 3
    logger.info(f"Sweep: source heights {sweep['source_heights']}, input sizes {sweep['input_sizes'] or 'model default'}"
                f"{'' if getattr(detector, 'dynamic_input', False) else ' (ignored: this model has a fixed input size)'}, "
                f"confidence {sweep['conf_thresholds']}; {total} frames per pass on target '{args.target}'...")
    t_sweep = time.perf_counter()
    run = run_passes(detector=detector, model_path=model_path, target_name=args.target, device=device, det_cfg=det_cfg,
                     sweep=sweep, video_path=video_path, source_info=source, frames=total, warmup=args.warmup,
                     default_conf=default_conf, max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb, sample_count=samples_n,
                     simulated=simulated)
    note = resolve_note(args, model_arg)
    report, records, files = build_sweep_report(
        run=run, sweep=sweep, device=target_cfg.device_profile(args.target), detector=detector, model_path=model_path,
        target_name=args.target, target_info=target_info, det_cfg=det_cfg, frames_requested=args.frames, warmup=args.warmup,
        cold_start=cold_start, notes=note)
    best = sweep_mod.best_entry(report)
    logger.info(f"Sweep done in {time.perf_counter() - t_sweep:.1f} s: {len(report['sweep']['configs'])} configurations; "
                f"best {best['label']} ({report['sweep']['best']['reason']})")

    # Metrics of the best configuration in the format of the single-configuration run (legacy metrics files, web app)
    bp = next(p for p in run.passes if p.source["label"] == best["source"]["label"] and p.size == (best["input"]["h"], best["input"]["w"]))
    _, stats, tot = detection_view(bp.records, bp.frames, best["conf"])
    metrics = {**bp.metrics, "detection_stats": stats, "total_detections": tot, "notes": note, "cold_start": cold_start,
               "best_config": sweep_mod.best_summary(report)}
    print_summary_table(metrics)
    print_sweep_table(report)
    if not args.no_report:
        order = [e["id"] for e in report["sweep"]["configs"]]
        all_frames = [r for cid in order for r in records[cid]]
        samples = [(int(rel.rsplit("_f", 1)[-1][:-4]), files[rel]) for rel in best.get("samples") or []]
        write_report_bundle(report, all_frames, samples, resolve(args.report_root), link_samples=args.link_samples,
                            back_link=(args.back_link, "back to suite") if args.back_link else None, sweep_files=files)
        metrics["report_id"] = report["report_id"]
        metrics["report_overall"] = report["verdict"]["overall"]
        logger.info(f"REPORT {report['report_id']}")
    if not args.no_metrics_files:
        json_path, csv_path = save_benchmark_report(metrics, PROJECT_ROOT / "results/metrics")
        logger.info(f"Saved JSON metrics to: {json_path}")
        logger.info(f"Saved CSV metrics to:  {csv_path}")
    return metrics


def run_many(args) -> list:
    """Benchmarks each model in its own process so peak-RAM numbers are not polluted by the previous model."""
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, model in enumerate(args.model):
            logger.info(f"===== [{i + 1}/{len(args.model)}] {Path(model).name} =====")
            out_json = Path(tmp) / f"run_{i}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), *forwarded_args(args, [model]),
                   "--summary-json", str(out_json)]
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if out_json.exists():
                runs.extend(json.loads(out_json.read_text(encoding="utf-8"))["runs"])
            else:
                runs.append({"target": args.target, "model": Path(model).name,
                             "error": f"benchmark process exited with code {res.returncode}"})
    return runs

def main():
    args = parse_args()
    target_cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")

    # If user requests target listing
    if args.list_targets:
        target_cfg.list_targets()
        return

    # 1. Dispatch to Docker if --docker flag is enabled
    if args.docker:
        if not is_docker_daemon_running():
            logger.error("Docker daemon is not running. Please start Docker Desktop or run natively without --docker.")
            sys.exit(1)

        if args.notes_file:  # the container only sees the project folder (bind mounts): pass a project-relative path
            try:
                args.notes_file = resolve(args.notes_file).resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                logger.error("--notes-file must be inside the project folder when used with --docker")
                sys.exit(2)
        sub_args = forwarded_args(args, args.model)
        if args.summary_json:
            sub_args.extend(["--summary-json", args.summary_json])
        temp_notes = None
        if args.notes is not None:  # free text must not go through the docker command line: hand it over as a file
            temp_notes = write_temp_notes(args)
            sub_args.extend(["--notes-file", temp_notes.relative_to(PROJECT_ROOT).as_posix()])
        docker_cmd = build_docker_run_command(
            target=args.target,
            script_name="run_benchmark.py",
            script_args=sub_args,
            project_root=PROJECT_ROOT,
            max_ram_mb=args.max_ram_mb
        )
        logger.info(f"Dispatching execution to Docker container for target '{args.target}'...")
        logger.info(f"Docker Command: {' '.join(docker_cmd)}")

        try:
            res = subprocess.run(docker_cmd, cwd=str(PROJECT_ROOT))
        finally:
            if temp_notes:
                temp_notes.unlink(missing_ok=True)
        sys.exit(res.returncode)

    # 2. Native / In-Container execution
    if len(args.model) > 1:
        runs = run_many(args)
        print_comparison_table(runs)
        summary = args.summary_json or f"results/metrics/comparison_{args.target}_{datetime.now():%Y%m%d_%H%M%S}.json"
        logger.info(f"Saved comparison to: {save_comparison_report(runs, resolve(summary), csv_too=not args.summary_json)}")
        if all("error" in r for r in runs):
            sys.exit(1)
        return

    try:
        metrics = run_single(args, target_cfg, args.model[0])
    except Exception as e:
        logger.error(f"Benchmark failed for {Path(args.model[0]).name}: {e}")
        if args.summary_json:  # lets callers (web app, matrix runner) show the failure instead of an empty result
            save_comparison_report([{"target": args.target, "model": Path(args.model[0]).name, "error": str(e),
                                     "notes": resolve_note(args, args.model[0])}],
                                   resolve(args.summary_json))
        sys.exit(1)
    if args.summary_json:
        save_comparison_report([metrics], resolve(args.summary_json))

if __name__ == "__main__":
    main()
