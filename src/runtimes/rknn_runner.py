from typing import Optional
import numpy as np
from src.runtimes.base import BaseDetector, DetectionResult
from src.utils.logger import get_logger

logger = get_logger("RKNNRunner")

class RKNNTargetDetector(BaseDetector):
    """
    Adapter for Rockchip RK3588 NPU using RKNN runtime (.rknn).
    Requires rknn-toolkit-lite2 inside the target container/host.
    """
    def __init__(self, model_path: str, config: Optional[dict] = None):
        super().__init__(model_path, config)
        self.rknn = None
        self.load_model()

    def load_model(self) -> None:
        try:
            from rknnlite.api import RKNNLite
            logger.info(f"Initializing RK3588 RKNN engine from: {self.model_path}")
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model {self.model_path}")
            self.rknn.init_runtime()
        except ImportError:
            logger.warning(
                "rknnlite not found in this environment. "
                "RKNN runner is intended for execution inside the RK3588 target container."
            )

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.rknn is None:
            raise RuntimeError("RKNN runtime is not initialized. Run inside the rk3588 container.")
        return DetectionResult()
