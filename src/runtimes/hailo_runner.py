from typing import Optional
import numpy as np
from src.runtimes.base import BaseDetector, DetectionResult
from src.utils.logger import get_logger

logger = get_logger("HailoRunner")

class HailoTargetDetector(BaseDetector):
    """
    Adapter for Hailo-10H M.2 AI module using HailoRT (.hef).
    Requires hailo_platform inside the target container/host.
    """
    def __init__(self, model_path: str, config: Optional[dict] = None):
        super().__init__(model_path, config)
        self.device = None
        self.network_group = None
        self.load_model()

    def load_model(self) -> None:
        try:
            from hailo_platform import VDevice, HEF
            logger.info(f"Initializing Hailo-10H HEF model from: {self.model_path}")
            self.hef = HEF(self.model_path)
            self.device = VDevice()
            self.network_group = self.device.configure(self.hef)[0]
        except ImportError:
            logger.warning(
                "hailo_platform not found in this environment. "
                "Hailo runner is intended for execution on hardware equipped with HailoRT."
            )

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.network_group is None:
            raise RuntimeError("HailoRT runtime is not initialized. Run inside the hailo10h container.")
        return DetectionResult()
