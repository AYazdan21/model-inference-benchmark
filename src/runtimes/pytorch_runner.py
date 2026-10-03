import time
from typing import Optional
import numpy as np
from src.runtimes.base import BaseDetector, Detection, DetectionResult
from src.utils.logger import get_logger

logger = get_logger("PyTorchRunner")

class PyTorchDetector(BaseDetector):
    def __init__(self, model_path: str, config: Optional[dict] = None):
        super().__init__(model_path, config)
        self.model = None
        self.output_format = "ultralytics"
        self.dynamic_input = True  # Ultralytics resizes to any imgsz that is a multiple of the model stride
        self.class_names = {}
        self.device = self.config.get("inference", {}).get("device", "cpu")
        self.conf = self.config.get("model", {}).get("conf_threshold", 0.35)
        self.iou = self.config.get("model", {}).get("iou_threshold", 0.45)
        self.input_size = tuple(self.config.get("model", {}).get("input_size", [640, 640]))
        self.load_model()

    def load_model(self) -> None:
        try:
            from ultralytics import YOLO
            num_threads = self.config.get("inference", {}).get("num_threads")
            if num_threads:
                import torch
                torch.set_num_threads(int(num_threads))
            logger.info(f"Loading YOLO model from: {self.model_path} onto {self.device}")
            self.model = YOLO(self.model_path)
            self.class_names = dict(getattr(self.model, "names", None) or {})
        except ImportError:
            raise ImportError("ultralytics is required for PyTorchDetector. Run: pip install ultralytics")

    def predict(self, frame: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        
        # Inference using YOLO
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )
        
        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0

        detections = []
        preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0

        if results and len(results) > 0:
            res = results[0]
            # Speed dict usually contains 'preprocess', 'inference', 'postprocess' in ms
            if hasattr(res, "speed") and isinstance(res.speed, dict):
                preprocess_ms = res.speed.get("preprocess", 0.0)
                inference_ms = res.speed.get("inference", 0.0)
                postprocess_ms = res.speed.get("postprocess", 0.0)
            
            boxes = res.boxes
            names = res.names
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                    detections.append(Detection(
                        box=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    ))

        return DetectionResult(
            detections=detections,
            latency_ms=total_latency_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms
        )
