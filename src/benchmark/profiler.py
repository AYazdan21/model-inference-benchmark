import os
import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import cv2
import numpy as np
import psutil
from src.runtimes.base import Detection, DetectionResult
from src.utils.resource_limiter import MemoryTracker

if TYPE_CHECKING:
    from src.simulation import DeviceProfile, ModelProfile

SAMPLE_MAX_WIDTH = 960   # kept sample frames are downscaled to this width to stay light on RAM
SAMPLE_COUNT = 3         # frames with the most detections are kept for the report images


def _stats(values, keys=("mean", "std", "min", "p50", "p90", "p95", "p99", "max")) -> Dict[str, float]:
    """Summary statistics of a list of numbers, rounded for reports."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: 0.0 for k in keys}
    table = {
        "mean": np.mean(arr), "std": np.std(arr), "min": np.min(arr), "max": np.max(arr),
        "p50": np.percentile(arr, 50), "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95), "p99": np.percentile(arr, 99),
    }
    return {k: round(float(table[k]), 2) for k in keys}


class BenchmarkProfiler:
    def __init__(
        self,
        target_name: str,
        model_name: str,
        memory_tracker: Optional[MemoryTracker] = None,
        device: Optional["DeviceProfile"] = None,
        model_profile: Optional["ModelProfile"] = None,
        sample_count: int = SAMPLE_COUNT,
    ):
        self.target_name = target_name
        self.model_name = model_name
        self.memory_tracker = memory_tracker or MemoryTracker()
        self.device = device
        self.model_profile = model_profile
        self.sample_count = max(0, sample_count)
        self.sim_latencies: List[float] = []
        self.sim_inference_ms: Optional[float] = None
        # CPU load is reported for this process, relative to the cores the target is allowed to use
        self.process = psutil.Process(os.getpid())
        self.cpu_cores = (device.cpu_cores if device and device.cpu_cores else None) or psutil.cpu_count() or 1
        self.total_latencies: List[float] = []
        self.preprocess_latencies: List[float] = []
        self.inference_latencies: List[float] = []
        self.postprocess_latencies: List[float] = []
        self.decode_latencies: List[float] = []
        self.cpu_usages: List[float] = []
        self.total_detections: int = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        # One record per frame (exported as frames.csv) and the best frames for report images
        self.frame_records: List[Dict[str, Any]] = []
        self.top_frames: List[Dict[str, Any]] = []
        self._class_confs: Dict[str, List[float]] = {}
        self._all_confs: List[float] = []

    def start(self) -> None:
        self.process.cpu_percent(interval=None)  # prime the counter
        self.start_time = time.perf_counter()

    def record_frame(
        self,
        result: DetectionResult,
        frame_index: Optional[int] = None,
        decode_ms: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        self.total_latencies.append(result.latency_ms)
        self.preprocess_latencies.append(result.preprocess_ms)
        self.inference_latencies.append(result.inference_ms)
        self.postprocess_latencies.append(result.postprocess_ms)
        if decode_ms is not None:
            self.decode_latencies.append(decode_ms)
        self.total_detections += len(result.detections)
        if result.sim_latency_ms is not None:
            self.sim_latencies.append(result.sim_latency_ms)
            self.sim_inference_ms = result.sim_inference_ms

        # Track process CPU as a share of the target's cores
        cpu = min(100.0, self.process.cpu_percent(interval=None) / self.cpu_cores)
        self.cpu_usages.append(cpu)

        # Track hardware memory limits (RAM & VRAM)
        ram_mb, _ = self.memory_tracker.sample()

        idx = len(self.frame_records) if frame_index is None else frame_index
        class_counts: Dict[str, int] = {}
        for det in result.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            self._class_confs.setdefault(det.class_name, []).append(det.confidence)
            self._all_confs.append(det.confidence)
        self.frame_records.append({
            "frame_index": idx,
            "decode_ms": round(decode_ms, 3) if decode_ms is not None else None,
            "host_latency_ms": round(result.latency_ms, 3),
            "preprocess_ms": round(result.preprocess_ms, 3),
            "inference_ms": round(result.inference_ms, 3),
            "postprocess_ms": round(result.postprocess_ms, 3),
            "sim_latency_ms": round(result.sim_latency_ms, 3) if result.sim_latency_ms is not None else None,
            "sim_inference_ms": round(result.sim_inference_ms, 3) if result.sim_inference_ms is not None else None,
            "detections": len(result.detections),
            "class_counts": class_counts,
            "max_confidence": round(max((d.confidence for d in result.detections), default=0.0), 4),
            "rss_mb": round(ram_mb, 1),
            "cpu_percent": round(cpu, 1),
        })
        if frame is not None:
            self._keep_sample(idx, frame, result)

    def _keep_sample(self, idx: int, frame: np.ndarray, result: DetectionResult) -> None:
        """Keeps the sample_count frames with the most detections (downscaled, so RAM stays flat)."""
        if self.sample_count == 0:
            return
        n = len(result.detections)
        if len(self.top_frames) >= self.sample_count and n <= min(t["count"] for t in self.top_frames):
            return
        h, w = frame.shape[:2]
        scale = min(1.0, SAMPLE_MAX_WIDTH / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
        scaled = DetectionResult(
            detections=[Detection(box=[c * scale for c in d.box], confidence=d.confidence,
                                  class_id=d.class_id, class_name=d.class_name) for d in result.detections],
            latency_ms=result.latency_ms, preprocess_ms=result.preprocess_ms,
            inference_ms=result.inference_ms, postprocess_ms=result.postprocess_ms,
            sim_latency_ms=result.sim_latency_ms, sim_inference_ms=result.sim_inference_ms,
        )
        self.top_frames.append({"frame_index": idx, "count": n, "frame": small, "result": scaled})
        self.top_frames.sort(key=lambda t: (-t["count"], t["frame_index"]))
        del self.top_frames[self.sample_count:]

    def finish(self) -> Dict[str, Any]:
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        num_frames = len(self.total_latencies)

        if num_frames == 0:
            return {"error": "No frames recorded"}

        lats = np.array(self.total_latencies)
        inf_lats = np.array(self.inference_latencies)
        mem_summary = self.memory_tracker.get_summary()

        # Throughput counts processing time only (video decode / benchmark bookkeeping excluded)
        busy_seconds = float(np.sum(lats)) / 1000.0
        fps = num_frames / busy_seconds if busy_seconds > 0 else 0.0
        wall_fps = num_frames / elapsed if elapsed > 0 else 0.0

        metrics = {
            "target": self.target_name,
            "model": self.model_name,
            "frames_processed": num_frames,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_fps": round(fps, 2),
            "wall_fps": round(wall_fps, 2),
            "latency_ms": _stats(lats),
            "inference_only_ms": {
                "mean": round(float(np.mean(inf_lats)), 2),
                "p50": round(float(np.percentile(inf_lats, 50)), 2),
                "p95": round(float(np.percentile(inf_lats, 95)), 2),
            },
            "stages_ms": {
                "preprocess": _stats(self.preprocess_latencies, ("mean", "p95")),
                "inference": _stats(self.inference_latencies, ("mean", "p95")),
                "postprocess": _stats(self.postprocess_latencies, ("mean", "p95")),
                "decode": _stats(self.decode_latencies, ("mean", "p95")) if self.decode_latencies else None,
            },
            "system_resources": {
                "avg_cpu_percent": round(float(np.mean(self.cpu_usages)), 1) if self.cpu_usages else 0.0,
                "peak_cpu_percent": round(float(np.max(self.cpu_usages)), 1) if self.cpu_usages else 0.0,
                "cpu_cores": self.cpu_cores,
                **mem_summary
            },
            "total_detections": self.total_detections,
            "detection_stats": self._detection_stats(num_frames),
            "model_profile": self.model_profile.to_dict() if self.model_profile else None,
            "simulated": self._simulated_summary(),
        }
        return metrics

    def _detection_stats(self, num_frames: int) -> Dict[str, Any]:
        counts = [r["detections"] for r in self.frame_records]
        with_dets = sum(1 for c in counts if c > 0)
        confs = self._all_confs
        return {
            "total": self.total_detections,
            "mean_per_frame": round(self.total_detections / num_frames, 3),
            "max_per_frame": max(counts) if counts else 0,
            "frames_with_detections": with_dets,
            "frames_with_detections_pct": round(100.0 * with_dets / num_frames, 1),
            "per_class": {
                name: {"count": len(c), "mean_conf": round(float(np.mean(c)), 3), "max_conf": round(float(np.max(c)), 3)}
                for name, c in sorted(self._class_confs.items(), key=lambda kv: -len(kv[1]))
            },
            "confidence": {
                "mean": round(float(np.mean(confs)), 3) if confs else None,
                "min": round(float(np.min(confs)), 3) if confs else None,
                "max": round(float(np.max(confs)), 3) if confs else None,
            },
        }

    def _simulated_summary(self) -> Optional[Dict[str, Any]]:
        if not self.device or not self.sim_latencies:
            return None
        sim = np.array(self.sim_latencies)
        mean = float(np.mean(sim))
        cpu_side = [r["sim_latency_ms"] - (r["sim_inference_ms"] or 0.0)
                    for r in self.frame_records if r["sim_latency_ms"] is not None]
        return {
            "device": self.device.device,
            "compute_unit": self.device.compute_unit,
            "runtime": self.device.runtime,
            "est_fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
            "latency_ms": {
                "mean": round(mean, 2),
                "p50": round(float(np.percentile(sim, 50)), 2),
                "p90": round(float(np.percentile(sim, 90)), 2),
                "p95": round(float(np.percentile(sim, 95)), 2),
                "p99": round(float(np.percentile(sim, 99)), 2),
                "min": round(float(np.min(sim)), 2),
                "max": round(float(np.max(sim)), 2),
            },
            "inference_ms": round(self.sim_inference_ms, 2) if self.sim_inference_ms is not None else None,
            "cpu_side_ms": round(float(np.mean(cpu_side)), 2) if cpu_side else None,
            "cpu_scale": round(self.device.cpu_scale, 3),
            "reference": {
                "model": self.device.ref_model,
                "gflops": self.device.ref_gflops,
                "latency_ms": self.device.ref_latency_ms,
                "source": self.device.ref_source,
            },
        }
