from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None

@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    # Estimated timings on the simulated target device (None when not simulated)
    sim_latency_ms: Optional[float] = None
    sim_inference_ms: Optional[float] = None

class BaseDetector(ABC):
    """Abstract base class for all hardware-specific detection runners."""

    # True when the model accepts input sizes other than its default (set by the runtimes that can tell)
    dynamic_input: bool = False

    def __init__(self, model_path: str, config: Optional[dict] = None):
        self.model_path = model_path
        self.config = config or {}

    def set_conf(self, conf: float) -> None:
        """Confidence threshold used by the next predict() calls (benchmark sweeps run once at the lowest one)."""
        self.conf = float(conf)

    def set_input_size(self, size: tuple) -> None:
        """Model input (height, width) for the next predict() calls. Only models with a dynamic input accept it."""
        if not self.dynamic_input:
            raise ValueError("this model has a fixed input size")
        self.input_size = (int(size[0]), int(size[1]))

    @abstractmethod
    def load_model(self) -> None:
        """Load weights and initialize engine/session."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """Execute detection on a single BGR image/frame."""
        pass

    def warmup(self, num_iterations: int = 5, input_size: tuple = (640, 640)) -> None:
        """Execute dummy inferences to warm up caches and GPU/NPU kernels."""
        dummy_frame = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.predict(dummy_frame)
