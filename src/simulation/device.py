"""
Analytical edge-device latency model.

The model really runs on the host (for real detections). Its latency on the target device
is estimated from a published calibration anchor for that device:

    sim_inference_ms = max(min_inference_ms, anchor_latency_ms * model_gflops / anchor_gflops)
    sim_pre_post_ms  = host_pre_post_ms * host_cpu_score / device_cpu_score

i.e. the device's *effective* throughput on a YOLO-class network is taken from the anchor and
scaled by compute cost, and the CPU-side work (resize, normalize, NMS) is scaled by a
single-core CPU score ratio. Expect roughly +/-30-50% vs. real hardware: good enough to rank
models and to see which ones cannot reach real time, not a replacement for on-device tests.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from src.simulation.model_profile import ModelProfile


@dataclass
class DeviceProfile:
    target: str
    device: str
    simulate: bool
    compute_unit: str = "cpu"
    runtime: str = ""
    cpu_cores: Optional[int] = None
    cpu_single_core_score: Optional[float] = None
    host_cpu_single_core_score: Optional[float] = None
    min_inference_ms: float = 0.0
    ref_model: str = ""
    ref_gflops: Optional[float] = None
    ref_latency_ms: Optional[float] = None
    ref_source: str = ""

    @classmethod
    def from_target(cls, target_name: str, target_info: Dict[str, Any], simulation_cfg: Optional[Dict[str, Any]] = None) -> "DeviceProfile":
        hw = target_info.get("hardware") or {}
        ref = hw.get("reference") or {}
        host = (simulation_cfg or {}).get("host") or {}
        return cls(
            target=target_name,
            device=hw.get("device", target_info.get("description", target_name)),
            simulate=bool(hw.get("simulate", bool(ref))),
            compute_unit=hw.get("compute_unit", "cpu"),
            runtime=hw.get("runtime", ""),
            cpu_cores=hw.get("cpu_cores"),
            cpu_single_core_score=hw.get("cpu_single_core_score"),
            host_cpu_single_core_score=host.get("cpu_single_core_score"),
            min_inference_ms=float(hw.get("min_inference_ms", 0.0)),
            ref_model=ref.get("model", ""),
            ref_gflops=ref.get("gflops"),
            ref_latency_ms=ref.get("latency_ms"),
            ref_source=ref.get("source", ""),
        )

    @property
    def cpu_scale(self) -> float:
        """Multiplier for host-measured CPU work (pre/post-processing) to target CPU time."""
        if self.host_cpu_single_core_score and self.cpu_single_core_score:
            return self.host_cpu_single_core_score / self.cpu_single_core_score
        return 1.0

    def can_estimate(self, model: ModelProfile) -> bool:
        return bool(self.simulate and self.ref_gflops and self.ref_latency_ms and model.gflops)

    def estimate_inference_ms(self, model: ModelProfile) -> Optional[float]:
        if not self.can_estimate(model):
            return None
        scaled = self.ref_latency_ms * (model.gflops / self.ref_gflops)
        return max(self.min_inference_ms, scaled)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cpu_scale"] = round(self.cpu_scale, 3)
        return d
