#!/usr/bin/env python3
"""
Model Inference Benchmark Live Stream Worker
Runs detection inference frame-by-frame and streams binary JPEG frames over stdout.
Supports bidirectional control commands via stdin (SEEK, PAUSE, RESUME, STOP).
Can run natively on the host or inside a hardware-constrained Docker container.
"""

import sys
import time
import argparse
import threading
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TargetConfig, load_yaml
from src.runtimes import get_detector
from src.video import VideoReader, VideoAnnotator
from src.utils.resource_limiter import setup_resource_limiter
from src.utils.logger import get_logger

logger = get_logger("LiveStreamWorker")

# Command state shared with stdin listener
class StreamControl:
    def __init__(self):
        self.lock = threading.Lock()
        self.paused = False
        self.stop_requested = False
        self.seek_frame: int | None = None

control = StreamControl()

def stdin_listener():
    """Reads control commands from stdin."""
    while not control.stop_requested:
        try:
            line = sys.stdin.readline()
            if not line:
                # Controller went away (app stopped / docker client killed): stop instead of running on orphaned
                with control.lock:
                    control.stop_requested = True
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split(":", 1)
            cmd = parts[0].lower()
            val = parts[1] if len(parts) > 1 else ""

            with control.lock:
                if cmd == "pause":
                    control.paused = True
                elif cmd == "resume":
                    control.paused = False
                elif cmd == "seek":
                    try:
                        control.seek_frame = int(val)
                    except ValueError:
                        pass
                elif cmd in ("stop", "exit", "quit"):
                    control.stop_requested = True
                    break
        except Exception:
            break

def main():
    parser = argparse.ArgumentParser(description="Live Stream Inference Worker")
    parser.add_argument("--target", default="x86-cpu", help="Target hardware platform")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="", help="Path to save output video (if saving)")
    parser.add_argument("--save-output", action="store_true", help="Save annotated output")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--max-ram-mb", type=int, default=None, help="RAM limit in MB")
    parser.add_argument("--max-vram-mb", type=int, default=None, help="VRAM limit in MB")
    parser.add_argument("--no-simulate", action="store_true", help="Show host timings only (no device latency estimation)")
    parser.add_argument("--no-pace", action="store_true",
                        help="Do not slow playback down to the simulated device's speed")
    args = parser.parse_args()

    # Start stdin listener thread
    listener_thread = threading.Thread(target=stdin_listener, daemon=True)
    listener_thread.start()

    writer = None
    reader = None

    try:
        det_cfg = load_yaml(PROJECT_ROOT / "configs" / "detection.yaml")
        det_cfg.setdefault("model", {})["conf_threshold"] = args.conf

        target_cfg = TargetConfig(PROJECT_ROOT / "targets.yaml")
        target_info = target_cfg.get_target(args.target)
        device = None if args.no_simulate else target_cfg.device_profile(args.target)

        reader = VideoReader(args.video)
        annotator = VideoAnnotator(target_name=args.target, device_name=device.device if device else "")
        detector = get_detector(args.target, args.model, det_cfg, device=device)
        mem_tracker = setup_resource_limiter(
            max_ram_mb=args.max_ram_mb or target_info.get("ram_limit_mb"),
            max_vram_mb=args.max_vram_mb or target_info.get("vram_limit_mb"),
        )

        if args.save_output and args.output:
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_file), fourcc, reader.fps, (reader.width, reader.height))

        total_frames = reader.total_frames
        fps = reader.fps or 30.0

        # Announce initialization complete
        init_msg = f"READY {total_frames} {fps:.1f}\n".encode("utf-8")
        sys.stdout.buffer.write(init_msg)
        sys.stdout.buffer.flush()

        while True:
            with control.lock:
                if control.stop_requested:
                    break
                target_seek = control.seek_frame
                control.seek_frame = None
                is_paused = control.paused

            if target_seek is not None:
                reader.seek_frame(target_seek)

            if is_paused:
                time.sleep(0.05)
                continue

            ret, frame, idx = reader.read_frame()
            if not ret or frame is None:
                # Video reached end
                sys.stdout.buffer.write(b"EOF\n")
                sys.stdout.buffer.flush()
                break

            t0 = time.perf_counter()
            result = detector.predict(frame)
            t1 = time.perf_counter()
            host_latency = (t1 - t0) * 1000.0

            mem_tracker.sample()

            # On a simulated target, fps / latency are the device's estimates
            simulated = result.sim_latency_ms is not None
            latency = result.sim_latency_ms if simulated else host_latency

            current_fps = 1000.0 / latency if latency > 0 else fps
            annotated = annotator.annotate(frame, result, idx, current_fps)

            if writer:
                writer.write(annotated)

            _, jpeg_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = jpeg_buf.tobytes()
            jpeg_len = len(jpeg_bytes)
            det_count = len(result.detections)

            # If the host is faster than the simulated device, hold the frame so playback runs at the device's speed
            if simulated and not args.no_pace:
                remaining_ms = latency - (time.perf_counter() - t0) * 1000.0
                if remaining_ms > 0:
                    time.sleep(remaining_ms / 1000.0)

            # Protocol: FRAME <frame_idx> <total_frames> <fps> <latency_ms> <det_count> <jpeg_len> <host_latency_ms> <simulated 0|1>\n<bytes>
            header = (f"FRAME {idx} {total_frames} {current_fps:.1f} {latency:.1f} {det_count} {jpeg_len} "
                      f"{host_latency:.1f} {int(simulated)}\n").encode("utf-8")
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()

    except Exception as e:
        err_msg = f"ERROR {str(e)}\n".encode("utf-8")
        sys.stdout.buffer.write(err_msg)
        sys.stdout.buffer.flush()
        sys.stderr.write(f"[LiveStreamWorker Error] {e}\n")
        sys.stderr.flush()
    finally:
        if reader:
            try:
                reader.release()
            except Exception:
                pass
        if writer:
            try:
                writer.release()
            except Exception:
                pass

if __name__ == "__main__":
    main()
