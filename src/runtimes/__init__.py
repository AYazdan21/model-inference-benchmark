import copy
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from src.runtimes.base import BaseDetector
from src.runtimes.pytorch_runner import PyTorchDetector
from src.runtimes.onnx_runner import ONNXDetector
from src.runtimes.jetson_runner import JetsonTensorRTDetector
from src.runtimes.rknn_runner import RKNNTargetDetector
from src.runtimes.hailo_runner import HailoTargetDetector

if TYPE_CHECKING:
    from src.simulation.device import DeviceProfile

# Device-native formats only run on real hardware with the vendor runtime installed
NATIVE_FORMATS = {
    ".engine": JetsonTensorRTDetector,
    ".trt": JetsonTensorRTDetector,
    ".rknn": RKNNTargetDetector,
    ".hef": HailoTargetDetector,
}

def get_detector(
    target_name: str,
    model_path: str,
    config: Optional[dict] = None,
    device: Optional["DeviceProfile"] = None,
) -> BaseDetector:
    """
    Builds the detector for a target.

    Native formats (.engine/.rknn/.hef) go to the vendor runtime. Portable formats (.onnx/.pt)
    run on the host CPU as a functional proxy; when `device` describes a simulated target, the
    detector is wrapped so every result also carries the estimated on-device latency.
    """
    ext = Path(model_path).suffix.lower()
    if ext in NATIVE_FORMATS:
        return NATIVE_FORMATS[ext](model_path, config)

    config = copy.deepcopy(config or {})
    if device is not None and device.cpu_cores:
        # Match the target's core count so threading behaviour resembles the device
        config.setdefault("inference", {}).setdefault("num_threads", device.cpu_cores)

    t_load = time.perf_counter()
    if ext == ".onnx":
        detector: BaseDetector = ONNXDetector(model_path, config)
    else:
        detector = PyTorchDetector(model_path, config)
    load_ms = (time.perf_counter() - t_load) * 1000.0

    if device is not None and device.simulate:
        from src.simulation import SimulatedDetector, profile_model
        input_size = tuple(getattr(detector, "input_size", None) or config.get("model", {}).get("input_size", [640, 640]))
        detector = SimulatedDetector(detector, device, profile_model(model_path, input_size))
    detector.load_ms = load_ms  # model load time only (excludes the static GFLOPs profiling)
    return detector

def apply_input_size(detector: BaseDetector, size: Optional[int]) -> Optional[str]:
    """
    Sets a square model input size (pixels, multiple of 32) on a model with a dynamic input.
    Models with a fixed input keep their size. Returns a message for the log, or None when size is None.
    """
    if not size:
        return None
    if not getattr(detector, "dynamic_input", False):
        h, w = getattr(detector, "input_size", None) or ("?", "?")
        return f"Input size {size} ignored: {Path(detector.model_path).name} has a fixed input of {h}x{w}"
    detector.set_input_size((int(size), int(size)))  # simulated detectors also re-estimate the device latency
    return f"Model input size set to {size}x{size}"

__all__ = [
    "BaseDetector",
    "PyTorchDetector",
    "ONNXDetector",
    "JetsonTensorRTDetector",
    "RKNNTargetDetector",
    "HailoTargetDetector",
    "get_detector",
    "apply_input_size",
    "NATIVE_FORMATS",
]
