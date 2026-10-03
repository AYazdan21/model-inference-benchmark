from typing import Optional
import numpy as np

from src.runtimes.base import BaseDetector, DetectionResult
from src.simulation.device import DeviceProfile
from src.simulation.model_profile import ModelProfile, profile_model
from src.utils.logger import get_logger

logger = get_logger("DeviceSimulator")


class SimulatedDetector(BaseDetector):
    """
    Runs the real model on the host (functional proxy, so detections are genuine) and
    annotates every result with the latency the target device is estimated to need.
    """

    def __init__(self, inner: BaseDetector, device: DeviceProfile, model_profile: ModelProfile):
        super().__init__(inner.model_path, inner.config)
        self.inner = inner
        self.device = device
        self.model_profile = model_profile
        self.sim_inference_ms: Optional[float] = device.estimate_inference_ms(model_profile)

        if self.sim_inference_ms is None:
            reason = model_profile.error or "no calibration anchor / GFLOPs unknown"
            logger.warning(f"Cannot estimate '{model_profile.name}' on {device.device}: {reason}. Reporting host timings only.")
        else:
            logger.info(
                f"Simulating {device.device} [{device.runtime}]: {model_profile.name} "
                f"{model_profile.gflops} GFLOPs -> est. {self.sim_inference_ms:.1f} ms inference "
                f"(anchor {device.ref_model} {device.ref_gflops} GFLOPs = {device.ref_latency_ms} ms; "
                f"CPU-side work x{device.cpu_scale:.2f})"
            )

    def load_model(self) -> None:
        pass  # inner detector is already loaded

    @property
    def dynamic_input(self) -> bool:
        return bool(getattr(self.inner, "dynamic_input", False))

    @property
    def input_size(self):
        return getattr(self.inner, "input_size", None)

    def set_conf(self, conf: float) -> None:
        self.inner.set_conf(conf)

    def set_input_size(self, size: tuple) -> None:
        """Changes the model input size and re-profiles the GFLOPs, so the device estimate follows the new size."""
        self.inner.set_input_size(size)
        self.model_profile = profile_model(self.model_path, tuple(size))
        self.sim_inference_ms = self.device.estimate_inference_ms(self.model_profile)

    @property
    def output_format(self) -> Optional[str]:
        return getattr(self.inner, "output_format", None)

    @property
    def class_names(self) -> dict:
        return getattr(self.inner, "class_names", None) or {}

    def predict(self, frame: np.ndarray) -> DetectionResult:
        result = self.inner.predict(frame)
        if self.sim_inference_ms is not None:
            host_cpu_ms = max(0.0, result.latency_ms - result.inference_ms)
            result.sim_inference_ms = self.sim_inference_ms
            result.sim_latency_ms = host_cpu_ms * self.device.cpu_scale + self.sim_inference_ms
        return result
