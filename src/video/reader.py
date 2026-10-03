from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

class VideoReader:
    def __init__(self, source: str | int | Path):
        self.source = str(source) if isinstance(source, Path) else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = (self.total_frames / self.fps) if self.fps > 0 else 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Reads a single frame and returns (success, frame, frame_idx)."""
        ret, frame = self.cap.read()
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame, idx

    def seek_frame(self, target_frame: int) -> bool:
        """Seeks video to a specific frame number."""
        if not self.cap or not self.cap.isOpened():
            return False
        target = max(0, min(int(target_frame), max(0, self.total_frames - 1)))
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    def seek_seconds(self, seconds: float) -> bool:
        """Seeks video by relative or absolute seconds."""
        target_frame = int(seconds * self.fps)
        return self.seek_frame(target_frame)

    def get_current_frame(self) -> int:
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = 0
        while self.cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
