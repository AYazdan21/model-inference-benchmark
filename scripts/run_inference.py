import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.config import TargetConfig, load_yaml
from src.runtimes import get_detector
from src.video import VideoReader, VideoAnnotator
from src.utils.resource_limiter import setup_resource_limiter
from src.utils.docker_runner import build_docker_run_command, is_docker_daemon_running
from src.utils.logger import get_logger

logger = get_logger("Inference")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Detection Inference on Video")
    parser.add_argument("--target", type=str, default="x86-cpu", help="Target hardware profile from targets.yaml")
    parser.add_argument("--list-targets", action="store_true", help="List all available target platforms")
    parser.add_argument("--docker", action="store_true", help="Execute inference inside Docker container")
    parser.add_argument("--model", type=str, default="models/best-yolo11-seg.pt", help="Path to model weights")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file or camera index")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated video (optional)")
    parser.add_argument("--config", type=str, default="configs/detection.yaml", help="Path to detection config")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--max-ram-mb", type=int, default=None, help="RAM memory limit in MB")
    parser.add_argument("--max-vram-mb", type=int, default=None, help="GPU VRAM memory limit in MB")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video to disk")
    return parser.parse_args()

def main():
    args = parse_args()
    target_cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")

    if args.list_targets:
        target_cfg.list_targets()
        return

    # 1. Dispatch to Docker if --docker flag is enabled
    if args.docker:
        if not is_docker_daemon_running():
            logger.error("Docker daemon is not running. Please start Docker Desktop or run natively without --docker.")
            sys.exit(1)

        sub_args = ["--target", args.target]
        if args.model:
            sub_args.extend(["--model", args.model])
        if args.video:
            sub_args.extend(["--video", args.video])
        if args.output:
            sub_args.extend(["--output", args.output])
        if args.conf is not None:
            sub_args.extend(["--conf", str(args.conf)])
        if args.max_frames:
            sub_args.extend(["--max-frames", str(args.max_frames)])
        if args.max_ram_mb:
            sub_args.extend(["--max-ram-mb", str(args.max_ram_mb)])
        if args.max_vram_mb:
            sub_args.extend(["--max-vram-mb", str(args.max_vram_mb)])

        docker_cmd = build_docker_run_command(
            target=args.target,
            script_name="run_inference.py",
            script_args=sub_args,
            project_root=PROJECT_ROOT,
            max_ram_mb=args.max_ram_mb
        )
        logger.info(f"Dispatching inference to Docker container for target '{args.target}'...")
        logger.info(f"Docker Command: {' '.join(docker_cmd)}")

        res = subprocess.run(docker_cmd, cwd=str(PROJECT_ROOT))
        sys.exit(res.returncode)

    # 2. Native / In-Container execution
    if not args.video:
        logger.error("--video argument is required to run inference (e.g. --video videos/input/sample.mp4)")
        sys.exit(1)

    target_info = target_cfg.get_target(args.target)
    logger.info(f"Target selected: '{args.target}' ({target_info.get('description', '')})")

    det_cfg = load_yaml(PROJECT_ROOT / args.config)
    if args.conf is not None:
        det_cfg.setdefault("model", {})["conf_threshold"] = args.conf

    # Memory limits setup
    res_cfg = det_cfg.get("resources", {})
    max_ram_mb = args.max_ram_mb or res_cfg.get("max_ram_mb") or target_info.get("ram_limit_mb")
    max_vram_mb = args.max_vram_mb or res_cfg.get("max_vram_mb") or target_info.get("vram_limit_mb")
    memory_tracker = setup_resource_limiter(max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb)

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        sys.exit(1)

    logger.info(f"Initializing detector for target '{args.target}' with {model_path.name}...")
    detector = get_detector(args.target, str(model_path), det_cfg, device=target_cfg.device_profile(args.target))

    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    else:
        video_source = str(PROJECT_ROOT / args.video if not Path(args.video).is_absolute() else args.video)

    reader = VideoReader(video_source)
    annotator = VideoAnnotator(target_name=args.target)

    writer = None
    if args.output and not args.no_save:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, reader.fps, (reader.width, reader.height))
        logger.info(f"Saving output video to: {out_path}")
    elif args.no_save:
        logger.info("Video saving disabled (--no-save). Running in-memory.")

    logger.info(f"Starting inference (Source: {video_source}, Total Frames: {reader.total_frames})...")
    frame_count = 0
    try:
        for idx, frame in reader.frames(max_frames=args.max_frames):
            result = detector.predict(frame)
            memory_tracker.sample()
            # Report the simulated device's speed when available, otherwise the host's
            latency_ms = result.sim_latency_ms if result.sim_latency_ms is not None else result.latency_ms
            current_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            if writer or det_cfg.get("video", {}).get("save_output", False):
                annotated = annotator.annotate(frame, result, idx, current_fps)
                if writer:
                    writer.write(annotated)

            frame_count += 1
            if frame_count % 30 == 0 or frame_count == reader.total_frames:
                logger.info(
                    f"Frame {frame_count}/{reader.total_frames} | "
                    f"Detections: {len(result.detections)} | "
                    f"Latency: {latency_ms:.1f}ms ({current_fps:.1f} FPS"
                    f"{', simulated' if result.sim_latency_ms is not None else ''}) | "
                    f"Peak RAM: {memory_tracker.peak_ram_mb:.0f}MB"
                )
    finally:
        reader.release()
        if writer:
            writer.release()
        logger.info(f"Inference completed. Processed {frame_count} frames. Peak RAM: {memory_tracker.peak_ram_mb:.1f}MB")

if __name__ == "__main__":
    main()
