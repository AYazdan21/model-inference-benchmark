"""
Assembles the benchmark report (schema_version 1.0) from a profiler's metrics, the per-frame
records and the run context. The result is plain JSON-serialisable data; rendering lives in
`export.py` / `html_report.py`.
"""
import hashlib
import math
import os
import platform
import re
import socket
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.benchmark.notes import clean_note

SCHEMA_VERSION = "1.0"
DEFAULT_REQUIRED_FPS = 25.0
PACKAGES = ["onnxruntime", "torch", "ultralytics", "opencv-python-headless", "opencv-python", "numpy"]

METHOD_NOTE = (
    "The model really runs on the host inside a Docker service limited to the device's CPU cores and RAM, "
    "so detections, RAM and CPU-side behaviour are measured. On-device latency is estimated from a published "
    "benchmark of a YOLO-class model on that device (calibration anchor), scaled by the model's GFLOPs. "
    "Video decoding is timed separately and is not part of latency or FPS."
)
ACCURACY_DISCLAIMER = (
    "Device figures are estimates with roughly +/-30-50% error versus real hardware: use them to rank models "
    "and to spot models that cannot reach real time, not to sign off a deployment. Not modelled: INT8 "
    "quantisation accuracy loss, NPU operator fallbacks to CPU, thermal throttling and interface limits "
    "(e.g. Hailo on a Pi 5 PCIe x1 link)."
)


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def make_report_id(target: str, model_file: str, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return sanitize_id(f"{target}__{Path(model_file).stem}__{when:%Y%m%d_%H%M%S}")


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cgroup_limits() -> Dict[str, Any]:
    cpu_cores = None
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max" and period.isdigit() and int(period) > 0:
            cpu_cores = round(int(quota) / int(period), 2)
    mem_mb = None
    mem_max = _read_text("/sys/fs/cgroup/memory.max")
    if mem_max and mem_max.isdigit():
        mem_mb = round(int(mem_max) / 1024 ** 2)
    return {"cpu_limit_cores": cpu_cores, "memory_limit_mb": mem_mb}


def _cpu_model() -> Optional[str]:
    info = _read_text("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def collect_environment() -> Dict[str, Any]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return {
        "execution": "docker" if Path("/.dockerenv").exists() else "local",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cgroup": _cgroup_limits(),
        "packages": {
            "onnxruntime": versions.get("onnxruntime"),
            "torch": versions.get("torch"),
            "ultralytics": versions.get("ultralytics"),
            "opencv": versions.get("opencv-python-headless") or versions.get("opencv-python"),
            "numpy": versions.get("numpy"),
        },
    }


def _sha256_prefix(path: Path, chars: int = 16) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:chars]
    except OSError:
        return None


def _model_section(model_path: Path, detector: Any, metrics: Dict[str, Any]) -> Dict[str, Any]:
    inner = getattr(detector, "inner", detector)
    profile = metrics.get("model_profile")
    if not profile:
        try:
            from src.simulation import profile_model
            profile = profile_model(model_path, tuple(getattr(inner, "input_size", None) or (640, 640))).to_dict()
        except Exception:
            profile = {}
    names = getattr(inner, "class_names", None) or {}
    size = model_path.stat().st_size if model_path.exists() else 0
    input_size = getattr(inner, "input_size", None)
    return {
        "file": model_path.name,
        "format": model_path.suffix.lower(),
        "size_mb": round(size / 1024 ** 2, 2),
        "sha256": _sha256_prefix(model_path),
        "gflops": profile.get("gflops"),
        "params_m": profile.get("params_m"),
        "input_size": list(input_size) if input_size else None,
        "class_names": [names[k] for k in sorted(names)] if names else [],
        "output_format": getattr(inner, "output_format", None) or "unknown",
    }


def _target_section(target_name: str, target_info: Dict[str, Any], device: Any, sim: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    hw = target_info.get("hardware") or {}
    return {
        "key": target_name,
        "description": target_info.get("description", ""),
        "device": device.device if device else hw.get("device", target_name),
        "simulated": sim is not None,
        "compute_unit": hw.get("compute_unit", "cpu"),
        "runtime": hw.get("runtime", ""),
        "cpu_cores": hw.get("cpu_cores"),
        "ram_limit_mb": target_info.get("ram_limit_mb"),
        "vram_limit_mb": target_info.get("vram_limit_mb"),
        "calibration": {
            "model": device.ref_model, "gflops": device.ref_gflops,
            "latency_ms": device.ref_latency_ms, "source": device.ref_source,
        } if device and device.ref_gflops else None,
        "cpu_scale": round(device.cpu_scale, 3) if device else None,
        "method": METHOD_NOTE,
        "accuracy_disclaimer": ACCURACY_DISCLAIMER,
    }


def _realtime_section(metrics: Dict[str, Any], source: Dict[str, Any], required_fps_cfg: Optional[float]) -> Dict[str, Any]:
    required = required_fps_cfg or source.get("fps") or DEFAULT_REQUIRED_FPS
    required = float(required)
    sim = metrics.get("simulated")
    fps = sim["est_fps"] if sim else metrics["throughput_fps"]
    p95 = sim["latency_ms"]["p95"] if sim else metrics["latency_ms"]["p95"]
    factor = fps / required if required > 0 else 0.0
    budget = 1000.0 / required if required > 0 else None
    return {
        "required_fps": round(required, 2),
        "basis": "estimated device FPS" if sim else "host FPS (not simulated)",
        "device_realtime_factor": round(factor, 3),
        "realtime_capable": factor >= 1.0,
        "frame_budget_ms": round(budget, 2) if budget else None,
        "p95_within_budget": bool(budget and p95 <= budget),
        "analysed_frame_ratio": round(min(1.0, factor), 3),
        "analyse_every_n_frames": 1 if factor >= 1.0 else (math.ceil(1.0 / factor) if factor > 0 else None),
    }


def _resources_section(metrics: Dict[str, Any], target_cores: Optional[int]) -> Dict[str, Any]:
    res = metrics["system_resources"]
    limit, peak = res.get("ram_limit_mb"), res.get("peak_ram_mb", 0.0)
    return {
        "ram_limit_mb": limit,
        "peak_ram_mb": peak,
        "ram_headroom_mb": round(limit - peak, 1) if limit else None,
        "ram_limit_breached": res.get("ram_limit_breached", False),
        "cpu_cores": target_cores or res.get("cpu_cores"),
        "avg_cpu_percent": res.get("avg_cpu_percent"),
        "peak_cpu_percent": res.get("peak_cpu_percent"),
        "vram_limit_mb": res.get("vram_limit_mb"),
        "peak_vram_mb": res.get("peak_vram_mb"),
        "vram_limit_breached": res.get("vram_limit_breached", False),
    }


def _verdict(report: Dict[str, Any], synthetic: bool, sim_expected: bool) -> Dict[str, Any]:
    items: List[Dict[str, str]] = []
    add = lambda level, msg: items.append({"level": level, "message": msg})

    rt = report["realtime"]
    factor, req = rt["device_realtime_factor"], rt["required_fps"]
    where = "on the device (estimated)" if report["device_estimate"] else "on the host (measured)"
    if rt["realtime_capable"]:
        if rt["p95_within_budget"]:
            add("ok", f"Real-time capable {where}: {factor:.2f}x the required {req:g} FPS.")
        else:
            add("warn", f"Average speed meets {req:g} FPS {where} ({factor:.2f}x) but P95 latency exceeds the "
                        f"{rt['frame_budget_ms']} ms frame budget: occasional frames will be late.")
    else:
        n = rt["analyse_every_n_frames"]
        add("warn" if factor >= 0.5 else "fail",
            f"Not real-time {where}: {factor:.2f}x the required {req:g} FPS. "
            f"It can analyse about 1 of every {n} frames.")

    res = report["resources"]
    if res["ram_limit_mb"]:
        if res["ram_limit_breached"]:
            add("fail", f"RAM budget exceeded: peak {res['peak_ram_mb']} MB vs limit {res['ram_limit_mb']} MB.")
        elif res["ram_headroom_mb"] < 0.1 * res["ram_limit_mb"]:
            add("warn", f"RAM headroom is thin: peak {res['peak_ram_mb']} MB of {res['ram_limit_mb']} MB.")
        else:
            add("ok", f"Fits the RAM budget: peak {res['peak_ram_mb']} MB of {res['ram_limit_mb']} MB.")
    if res["vram_limit_mb"] and res["vram_limit_breached"]:
        add("fail", f"VRAM budget exceeded: peak {res['peak_vram_mb']} MB vs limit {res['vram_limit_mb']} MB.")

    fmt = report["model"]["output_format"]
    if fmt == "unsupported":
        add("warn", "Model output format not decoded by this harness (only YOLOv8/11 and end-to-end detection heads "
                    "are): latency and RAM are real, but 0 detections says nothing about what the model finds.")
    elif fmt == "unknown":
        add("warn", "Output format of this runtime is unknown; detection counts may not be comparable.")
    else:
        dets = report["detections"]
        if dets["total"] == 0 and synthetic:
            add("ok", "Synthetic noise frames were used: detection counts are not meaningful. Provide a video for that.")
        elif dets["total"] == 0:
            add("warn", f"No detections in {report['config']['frames_processed']} frames at confidence "
                        f"{report['config']['conf_threshold']}: check the video content or lower the threshold.")
        else:
            add("ok", f"Detections decoded ({fmt} head): {dets['total']} in {report['config']['frames_processed']} frames "
                      f"({dets['mean_per_frame']} per frame).")

    if sim_expected and not report["device_estimate"]:
        add("warn", "Device latency could not be estimated for this model (GFLOPs or calibration missing); "
                    "only host timings are reported.")

    levels = [i["level"] for i in items]
    overall = "fail" if "fail" in levels else ("warn" if "warn" in levels else "ok")
    return {"overall": overall, "items": items}


def build_report(
    *,
    metrics: Dict[str, Any],
    frame_records: List[Dict[str, Any]],
    target_name: str,
    target_info: Dict[str, Any],
    device: Any,
    detector: Any,
    model_path: Path,
    det_cfg: Dict[str, Any],
    frames_requested: int,
    warmup: int,
    source: Dict[str, Any],
    cold_start: Dict[str, Any],
    when: Optional[datetime] = None,
    notes: str = "",
) -> Dict[str, Any]:
    when = when or datetime.now()
    sim = metrics.get("simulated")
    model_cfg = det_cfg.get("model", {})
    required_fps_cfg = (det_cfg.get("benchmark") or {}).get("required_fps")
    stages = metrics["stages_ms"]

    report: Dict[str, Any] = {
        "report_id": make_report_id(target_name, model_path.name, when),
        "created": when.astimezone().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "notes": clean_note(notes),  # free text of the user for this model/device pair ("" when none)
        "environment": collect_environment(),
        "target": _target_section(target_name, target_info, device, sim),
        "model": _model_section(model_path, detector, metrics),
        "config": {
            "conf_threshold": model_cfg.get("conf_threshold"),
            "iou_threshold": model_cfg.get("iou_threshold"),
            "frames_requested": frames_requested,
            "frames_processed": metrics["frames_processed"],
            "warmup": warmup,
            "source": source,
        },
        "cold_start": cold_start,
        "host_performance": {
            "latency_ms": metrics["latency_ms"],
            "stages": stages,
            "throughput_fps": metrics["throughput_fps"],
            "wall_fps": metrics["wall_fps"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        },
        "device_estimate": {
            "inference_ms": sim["inference_ms"],
            "latency_ms": sim["latency_ms"],
            "est_fps": sim["est_fps"],
            "cpu_side_ms": sim["cpu_side_ms"],
        } if sim else None,
        "realtime": _realtime_section(metrics, source, required_fps_cfg),
        "resources": _resources_section(metrics, device.cpu_cores if device else None),
        "detections": metrics["detection_stats"],
    }
    report["verdict"] = _verdict(report, synthetic=source.get("type") == "synthetic",
                                 sim_expected=bool(device and device.simulate))
    report["frames_csv"] = "frames.csv"
    report["samples"] = []
    return report
