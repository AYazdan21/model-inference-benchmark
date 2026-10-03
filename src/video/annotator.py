from typing import List
import cv2
import numpy as np
from src.runtimes.base import DetectionResult

class VideoAnnotator:
    def __init__(self, target_name: str = "Unknown", device_name: str = ""):
        self.target_name = target_name
        self.device_name = device_name
        self.colors = [
            (0, 0, 255),    # Red
            (0, 165, 255),  # Orange
            (0, 255, 255),  # Yellow
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Magenta
        ]

    def annotate(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        frame_idx: int,
        fps: float
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw detections
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.box)
            color = self.colors[det.class_id % len(self.colors)]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text
            label = f"{det.class_name}: {det.confidence:.2f}"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - t_size[1] - 6),
                (x1 + t_size[0] + 4, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw Telemetry & Target info overlay
        if result.sim_latency_ms is not None:
            lines = [
                f"Target: {self.target_name} | Simulated: {self.device_name} | FPS: {fps:.1f} | "
                f"Latency: {result.sim_latency_ms:.1f}ms | Frame: {frame_idx}",
                f"Host actual: {result.latency_ms:.1f}ms | Est. device inference: {result.sim_inference_ms:.1f}ms",
            ]
        else:
            lines = [f"Target: {self.target_name} | FPS: {fps:.1f} | Latency: {result.latency_ms:.1f}ms | Frame: {frame_idx}"]
        cv2.rectangle(annotated, (10, 10), (w - 10, 14 + 28 * len(lines)), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (16, 32 + 28 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if i == 0 else (180, 200, 220),
                2 if i == 0 else 1,
                cv2.LINE_AA
            )

        return annotated
