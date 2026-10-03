"""
Execution engine of a benchmark sweep (see sweep.py for the idea).

One model is loaded once. For every (source resolution, model input size) pass it runs the frames at the lowest
confidence threshold; the configurations of the higher thresholds are derived from that pass without more inference.
Passes run reference-first (highest source resolution, largest input size), so the frames that are shown to the user
(the ones with the most detections in the reference configuration) are known as early as possible.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.benchmark import sweep as sw
from src.benchmark.profiler import BenchmarkProfiler
from src.benchmark.report_builder import _realtime_section, _resources_section, _verdict, build_report
from src.runtimes.base import DetectionResult
from src.utils.logger import get_logger
from src.utils.resource_limiter import MemoryTracker
from src.video import VideoReader

logger = get_logger("SweepRunner")

PROGRESS_EVERY = 5            # frames between "PROGRESS done/total" lines (parsed by the web app)
FRAME_CACHE_MB = 400          # decoded frames of one source resolution are kept for the next passes below this size
SAMPLE_WIDTH = 640            # sample images of the configurations are downscaled to this width
SAMPLE_QUALITY = 72
SAMPLE_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
WARMUP_ON_SIZE_CHANGE = 3     # warm-up inferences after the input size changed (dynamic-shape sessions re-plan)


# ---------------------------------------------------------------- frames

class FrameSource:
    """Frames of one video (or synthetic noise) at a chosen source resolution. Decoding is timed per frame; downscaling
    is not. Frames of a source resolution are kept for the following passes when they fit in FRAME_CACHE_MB."""

    def __init__(self, video_path: Optional[Path], frames: int, input_size_hint: Tuple[int, int]):
        self.video_path = video_path
        self.requested = frames
        self.n_frames: Optional[int] = None   # known after the first pass (a video can be shorter than requested)
        self.hint = input_size_hint
        self._cache_label: Optional[str] = None
        self._cache: List[Tuple[np.ndarray, Optional[float]]] = []
        self._cache_complete = False
        self._cache_bytes = 0

    @property
    def cache_mb(self) -> float:
        return self._cache_bytes / 1024 ** 2

    def _wanted(self) -> int:
        return self.n_frames if self.n_frames is not None else self.requested

    def frames(self, source: Dict[str, Any], size: Tuple[int, int], keep: bool) -> Iterator[Tuple[int, np.ndarray, Optional[float]]]:
        """(frame index, BGR frame at the source resolution, decode ms or None for synthetic frames)."""
        if self.video_path is None:
            for idx in range(self._wanted()):
                rng = np.random.default_rng(1000 + idx)
                yield idx, rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8), None
            return
        label = source["label"]
        if self._cache_label == label and self._cache_complete:
            for idx, (frame, dec) in enumerate(self._cache):
                yield idx, frame, dec
            return
        self._drop_cache()
        cap = keep and self._wanted() * source["width"] * source["height"] * 3 <= FRAME_CACHE_MB * 1024 ** 2
        if cap:
            self._cache_label = label
        reader = VideoReader(self.video_path)
        try:
            for idx in range(self._wanted()):
                t0 = time.perf_counter()
                ok, frame, _ = reader.read_frame()
                if not ok or frame is None:
                    break
                dec = (time.perf_counter() - t0) * 1000.0
                if not source["native"]:
                    frame = cv2.resize(frame, (source["width"], source["height"]), interpolation=cv2.INTER_AREA)
                if cap:
                    self._cache.append((frame, dec))
                    self._cache_bytes += frame.nbytes
                yield idx, frame, dec
            else:
                self._cache_complete = cap
        finally:
            reader.release()
            if not self._cache_complete:
                self._drop_cache()

    def _drop_cache(self) -> None:
        self._cache, self._cache_label, self._cache_complete, self._cache_bytes = [], None, False, 0


# ---------------------------------------------------------------- results

@dataclass
class PassResult:
    source: Dict[str, Any]
    size: Tuple[int, int]
    metrics: Dict[str, Any]                       # profiler.finish() (detections at the lowest confidence)
    records: List[Dict[str, Any]]                 # per-frame records (detections at the lowest confidence)
    frames: List[List[Tuple]]                     # normalised detections per frame at the lowest confidence
    gflops: Optional[float] = None
    params_m: Optional[float] = None


@dataclass
class SweepRun:
    video: str
    source: Dict[str, Any]
    sources: List[Dict[str, Any]]
    sizes: List[Tuple[int, int]]
    locked: bool
    default_conf: float
    min_conf: float
    confs: List[float]
    passes: List[PassResult]
    n_frames: int
    sample_frames: List[int]
    sample_images: Dict[Tuple[str, int], np.ndarray]  # (source label, frame index) -> small BGR frame


def _norm_dets(result: DetectionResult, frame: np.ndarray) -> List[Tuple]:
    h, w = frame.shape[:2]
    return [(d.box[0] / w, d.box[1] / h, d.box[2] / w, d.box[3] / h, float(d.confidence), int(d.class_id), str(d.class_name))
            for d in result.detections]


def detection_view(records: List[Dict[str, Any]], frames: List[List[Tuple]], conf: float):
    """Per-frame records, detection statistics and total detections as if the run had used `conf` (derived by filtering)."""
    out, counts, class_confs, all_confs = [], [], {}, []
    for rec, dets in zip(records, frames):
        kept = [d for d in dets if d[4] >= conf]
        cc: Dict[str, int] = {}
        for d in kept:
            cc[d[6]] = cc.get(d[6], 0) + 1
            class_confs.setdefault(d[6], []).append(d[4])
            all_confs.append(d[4])
        r = dict(rec)
        r["detections"] = len(kept)
        r["class_counts"] = cc
        r["max_confidence"] = round(max((d[4] for d in kept), default=0.0), 4)
        out.append(r)
        counts.append(len(kept))
    n = len(counts)
    total = sum(counts)
    with_dets = sum(1 for c in counts if c > 0)
    stats = {
        "total": total,
        "mean_per_frame": round(total / n, 3) if n else 0.0,
        "max_per_frame": max(counts) if counts else 0,
        "frames_with_detections": with_dets,
        "frames_with_detections_pct": round(100.0 * with_dets / n, 1) if n else 0.0,
        "per_class": {name: {"count": len(c), "mean_conf": round(float(np.mean(c)), 3), "max_conf": round(float(np.max(c)), 3)}
                      for name, c in sorted(class_confs.items(), key=lambda kv: -len(kv[1]))},
        "confidence": {"mean": round(float(np.mean(all_confs)), 3) if all_confs else None,
                       "min": round(float(np.min(all_confs)), 3) if all_confs else None,
                       "max": round(float(np.max(all_confs)), 3) if all_confs else None},
    }
    return out, stats, total


# ---------------------------------------------------------------- the passes

def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= SAMPLE_WIDTH:
        return frame.copy()
    return cv2.resize(frame, (SAMPLE_WIDTH, int(round(h * SAMPLE_WIDTH / w))), interpolation=cv2.INTER_AREA)


def render_sample(small: np.ndarray, dets: List[Tuple], caption: str) -> bytes:
    """JPEG of a sample frame with the boxes of one configuration and a caption line."""
    img = small.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2, conf, cls, name in dets:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        color = SAMPLE_COLORS[cls % len(SAMPLE_COLORS)]
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (p1[0], max(0, p1[1] - th - 5)), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, label, (p1[0] + 2, max(th, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, f"{caption} | {len(dets)} boxes", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, SAMPLE_QUALITY])
    return buf.tobytes() if ok else b""


def run_passes(*, detector, model_path: Path, target_name: str, device, det_cfg: Dict[str, Any], sweep: Dict[str, Any],
               video_path: Optional[Path], source_info: Dict[str, Any], frames: int, warmup: int, default_conf: float,
               max_ram_mb: Optional[int], max_vram_mb: Optional[int], sample_count: int, simulated: bool) -> SweepRun:
    """Runs every (source resolution, input size) pass and returns the raw material of the configurations."""
    from src.simulation import profile_model

    inner = getattr(detector, "inner", detector)
    dynamic = bool(getattr(detector, "dynamic_input", False))
    default_size = tuple(getattr(inner, "input_size", None) or (640, 640))
    sources = sw.resolve_sources(sweep["source_heights"], source_info["width"], source_info["height"]) \
        if video_path else [{"label": "native", "text": f"native ({default_size[1]}x{default_size[0]}, synthetic)",
                             "height": default_size[0], "width": default_size[1], "native": True}]
    sizes = sw.resolve_inputs(dynamic, default_size, sweep["input_sizes"])
    confs = sorted(set(sweep["conf_thresholds"]) | {default_conf})
    min_conf = confs[0]
    detector.set_conf(min_conf)

    order = [(s, z) for s in sources for z in sizes]
    n_passes = len(order)
    src = FrameSource(video_path, frames, default_size)
    total_units = n_passes * frames
    done_units = 0
    passes: List[PassResult] = []
    samples_idx: List[int] = []
    sample_images: Dict[Tuple[str, int], np.ndarray] = {}   # (source label, frame index) -> small BGR frame
    captured: set = set()                                    # source labels whose sample frames are in sample_images
    current_size = default_size
    passes_left = {s["label"]: sum(1 for ss, _ in order if ss["label"] == s["label"]) for s in sources}
    n_frames = frames

    for i, (source, size) in enumerate(order, 1):
        label = source["label"]
        print(f"SWEEP_PASS {i}/{n_passes} {label} {sw.input_short(size)}", flush=True)
        if dynamic and tuple(size) != tuple(current_size):
            detector.set_input_size(size)
            current_size = tuple(size)
            dummy = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            for _ in range(max(1, min(warmup, WARMUP_ON_SIZE_CHANGE))):
                detector.predict(dummy)
        prof = profile_model(model_path, tuple(size))   # GFLOPs at this input size (cached)

        tracker = MemoryTracker(max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb)
        profiler = BenchmarkProfiler(target_name=target_name, model_name=model_path.name, memory_tracker=tracker, device=device,
                                     model_profile=getattr(detector, "model_profile", None) if simulated else None,
                                     sample_count=0)
        keep = passes_left[label] > 1
        grab = set(samples_idx) if samples_idx and label not in captured else set()
        det_frames: List[List[Tuple]] = []
        profiler.start()
        for idx, frame, decode_ms in src.frames(source, size, keep):
            tracker.offset_mb = src.cache_mb
            result = detector.predict(frame)
            profiler.record_frame(result, frame_index=idx, decode_ms=decode_ms, frame=None)
            det_frames.append(_norm_dets(result, frame))
            if idx in grab:
                sample_images[(label, idx)] = _small(frame)
            done_units += 1
            if done_units % PROGRESS_EVERY == 0 or done_units == total_units:
                print(f"PROGRESS {done_units}/{total_units}", flush=True)
        metrics = profiler.finish()
        passes_left[label] -= 1
        if "error" in metrics:
            raise RuntimeError(f"{metrics['error']} (the video ended before the first frame)")
        if grab:
            captured.add(label)
        if src.n_frames is None:
            src.n_frames = n_frames = metrics["frames_processed"]
            total_units = n_passes * n_frames
        passes.append(PassResult(source=source, size=tuple(size), metrics=metrics, records=list(profiler.frame_records),
                                 frames=det_frames, gflops=prof.gflops, params_m=prof.params_m))

        if i == 1:  # the reference pass: its detections choose the frames every configuration is shown on
            counts = [len([d for d in dets if d[4] >= default_conf]) for dets in det_frames]
            samples_idx = sw.pick_sample_frames(counts, sample_count)
            wanted = set(samples_idx)
            if wanted:
                for idx, frame, _ in src.frames(source, size, keep):
                    if idx in wanted:
                        sample_images[(label, idx)] = _small(frame)
                captured.add(label)

    return SweepRun(video=video_path.name if video_path else "synthetic", source=source_info, sources=sources, sizes=sizes,
                    locked=not dynamic, default_conf=default_conf, min_conf=min_conf, confs=confs, passes=passes,
                    n_frames=n_frames, sample_frames=samples_idx, sample_images=sample_images)


# ---------------------------------------------------------------- configurations and report

def _device_estimate(sim: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sim:
        return None
    return {"inference_ms": sim["inference_ms"], "latency_ms": sim["latency_ms"], "est_fps": sim["est_fps"],
            "cpu_side_ms": sim["cpu_side_ms"]}


def build_entries(run: SweepRun, *, device, required_fps_cfg: Optional[float], output_format: str, synthetic: bool,
                  sim_expected: bool) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, bytes]]:
    """(configuration entries, per-frame records by config id, sample JPEGs by relative file name) for every pass x confidence."""
    ref = run.passes[0]
    ref_frames = sw.filter_frames(ref.frames, run.default_conf)
    ref_id = sw.config_id(ref.source["label"], ref.size, run.default_conf)
    entries: List[Dict[str, Any]] = []
    records: Dict[str, List[Dict[str, Any]]] = {}
    files: Dict[str, bytes] = {}
    measurable = output_format not in ("unsupported", "unknown")
    for p in run.passes:
        m = p.metrics
        for conf in run.confs:
            cid = sw.config_id(p.source["label"], p.size, conf)
            recs, stats, total = detection_view(p.records, p.frames, conf)
            filtered = sw.filter_frames(p.frames, conf)
            agree = sw.agreement(ref_frames, filtered) if measurable else None
            temporal = sw.temporal_consistency(filtered) if measurable else None
            metrics = {**m, "detection_stats": stats, "total_detections": total}
            host = {"latency_ms": m["latency_ms"], "stages": m["stages_ms"], "throughput_fps": m["throughput_fps"],
                    "wall_fps": m["wall_fps"], "elapsed_seconds": m["elapsed_seconds"]}
            est = _device_estimate(m.get("simulated"))
            rt = _realtime_section(metrics, run.source, required_fps_cfg)
            res = _resources_section(metrics, device.cpu_cores if device else None)
            pseudo = {"realtime": rt, "resources": res, "device_estimate": est, "model": {"output_format": output_format},
                      "detections": stats, "config": {"frames_processed": m["frames_processed"], "conf_threshold": conf}}
            entry = {
                "id": cid,
                "label": sw.config_text(p.source, p.size, conf, run.locked),
                "short": sw.config_short(p.source, p.size, conf),
                "source": {k: p.source[k] for k in ("label", "text", "width", "height", "native")},
                "input": {"h": p.size[0], "w": p.size[1], "label": sw.input_label(p.size), "locked": run.locked},
                "conf": conf, "gflops": p.gflops, "params_m": p.params_m,
                "host_performance": host, "device_estimate": est, "realtime": rt, "resources": res, "detections": stats,
                "verdict": _verdict(pseudo, synthetic=synthetic, sim_expected=sim_expected),
                "agreement": agree, "temporal": temporal, "stability": sw.stability_score(agree["f1"] if agree else None, temporal),
                "is_reference": cid == ref_id, "samples": [], "ratings": None,
            }
            for idx in run.sample_frames:
                small = run.sample_images.get((p.source["label"], idx))
                if small is None:
                    continue
                rel = f"samples/{sw.config_slug(cid)}_f{idx:06d}.jpg"
                dets = [d for d in p.frames[idx] if d[4] >= conf] if idx < len(p.frames) else []
                files[rel] = render_sample(small, dets, f"frame {idx} | {entry['short']}")
                entry["samples"].append(rel)
            entries.append(entry)
            records[cid] = [{"config_id": cid, "source": p.source["label"], "input": sw.input_label(p.size), "conf": conf, **r}
                            for r in recs]
    return entries, records, files


def build_sweep_report(*, run: SweepRun, sweep: Dict[str, Any], device, detector, model_path: Path, target_name: str,
                       target_info: Dict[str, Any], det_cfg: Dict[str, Any], frames_requested: int, warmup: int,
                       cold_start: Dict[str, Any], notes: str, store=None):
    """(report, per-frame records by config id, sample JPEGs by file name). The top-level sections of the report
    describe the best configuration."""
    inner = getattr(detector, "inner", detector)
    output_format = getattr(inner, "output_format", None) or "unknown"
    required_cfg = (det_cfg.get("benchmark") or {}).get("required_fps")
    entries, records, files = build_entries(
        run, device=device, required_fps_cfg=required_cfg, output_format=output_format,
        synthetic=run.video == "synthetic", sim_expected=bool(device and device.simulate))
    ref_pass = run.passes[0]
    _, ref_stats, ref_total = detection_view(ref_pass.records, ref_pass.frames, run.default_conf)
    ref_metrics = {**ref_pass.metrics, "detection_stats": ref_stats, "total_detections": ref_total}
    report = build_report(
        metrics=ref_metrics, frame_records=ref_pass.records, target_name=target_name, target_info=target_info, device=device,
        detector=detector, model_path=model_path, det_cfg=det_cfg, frames_requested=frames_requested, warmup=warmup,
        source=run.source, cold_start=cold_start, notes=notes)
    report["config"]["conf_threshold"] = run.default_conf
    report["sweep"] = {
        "schema": 1,
        "video": run.video,
        "dimensions": {
            "source": [{k: s[k] for k in ("label", "text", "width", "height", "native")} for s in run.sources],
            "input_sizes": [list(z) for z in run.sizes], "input_locked": run.locked,
            "conf_thresholds": run.confs, "default_conf": run.default_conf, "min_conf": run.min_conf,
            "requested": {k: sweep[k] for k in ("source_heights", "input_sizes", "conf_thresholds")},
        },
        "reference_config": next(e["id"] for e in entries if e["is_reference"]),
        "sample_frames": run.sample_frames,
        "configs": entries,
        "best": None,
        "ratings_signature": None,
        "notes": sw.SWEEP_NOTES,
    }
    sw.apply_ratings(report, store=store)
    return report, records, files
