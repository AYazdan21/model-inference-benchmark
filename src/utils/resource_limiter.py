import os
import shutil
import sys
import platform
from typing import Optional, Tuple
import psutil
from src.utils.logger import get_logger

logger = get_logger("ResourceLimiter")

def _gpu_visible() -> bool:
    return shutil.which("nvidia-smi") is not None or os.path.exists("/dev/nvidiactl")

def _get_torch(allow_import: bool = False):
    """
    torch, only if it is already imported (PyTorch runner) or a GPU is visible and `allow_import` is set.
    Importing torch just to look at VRAM adds ~500 MB of RSS, which would inflate every peak-RAM measurement.
    """
    torch = sys.modules.get("torch")
    if torch is None and allow_import and _gpu_visible():
        try:
            import torch
        except ImportError:
            return None
    return torch

class MemoryTracker:
    """Tracks RAM and VRAM usage during execution."""
    def __init__(self, max_ram_mb: Optional[int] = None, max_vram_mb: Optional[int] = None):
        self.max_ram_mb = max_ram_mb
        self.max_vram_mb = max_vram_mb
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = 0.0
        # RAM held by the benchmark harness itself (e.g. cached video frames of a sweep): not counted as the model's RAM
        self.offset_mb = 0.0

    def sample(self) -> Tuple[float, float]:
        """
        Samples current RAM and VRAM usage.
        Returns (current_ram_mb, current_vram_mb).
        """
        # RAM usage (RSS)
        rss_bytes = self.process.memory_info().rss
        ram_mb = max(0.0, rss_bytes / (1024 ** 2) - self.offset_mb)
        if ram_mb > self.peak_ram_mb:
            self.peak_ram_mb = ram_mb

        # Check if RAM limit breached
        if self.max_ram_mb and ram_mb > self.max_ram_mb:
            logger.warning(
                f"[RAM EXCEEDED] Current RAM: {ram_mb:.1f} MB exceeds limit: {self.max_ram_mb} MB"
            )

        # VRAM usage
        vram_mb = 0.0
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            try:
                vram_bytes = torch.cuda.memory_allocated()
                vram_mb = vram_bytes / (1024 ** 2)
                if vram_mb > self.peak_vram_mb:
                    self.peak_vram_mb = vram_mb
                
                if self.max_vram_mb and vram_mb > self.max_vram_mb:
                    logger.warning(
                        f"[VRAM EXCEEDED] Current VRAM: {vram_mb:.1f} MB exceeds limit: {self.max_vram_mb} MB"
                    )
            except Exception:
                pass

        return ram_mb, vram_mb

    def get_summary(self) -> dict:
        return {
            "ram_limit_mb": self.max_ram_mb,
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "ram_limit_breached": bool(self.max_ram_mb and self.peak_ram_mb > self.max_ram_mb),
            "vram_limit_mb": self.max_vram_mb,
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "vram_limit_breached": bool(self.max_vram_mb and self.peak_vram_mb > self.max_vram_mb),
        }

def apply_vram_limit(max_vram_mb: Optional[int], device_id: int = 0) -> None:
    """Applies a hard upper cap on PyTorch GPU VRAM allocation."""
    if not max_vram_mb:
        return

    torch = _get_torch(allow_import=True)
    if torch is None or not torch.cuda.is_available():
        logger.info(f"VRAM budget {max_vram_mb} MB is not enforced: no CUDA GPU in this process "
                    f"(simulated GPU targets run the model on CPU; device speed is estimated).")
        return

    try:
        total_vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
        total_vram_mb = total_vram_bytes / (1024 ** 2)
        fraction = min(1.0, max(0.01, (max_vram_mb / total_vram_mb)))

        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        logger.info(
            f"VRAM Limit Enforced: {max_vram_mb} MB ({fraction * 100:.1f}% of total {total_vram_mb:.0f} MB on GPU {device_id})"
        )
    except Exception as e:
        logger.warning(f"Could not enforce VRAM limit: {e}")

def apply_ram_limit_os(max_ram_mb: Optional[int]) -> None:
    """Enforces and monitors RAM limits."""
    if not max_ram_mb:
        return
    logger.info(f"Target Hardware RAM Budget: {max_ram_mb} MB (enforced by Docker cgroups / monitored by profiler)")

def setup_resource_limiter(
    max_ram_mb: Optional[int] = None,
    max_vram_mb: Optional[int] = None,
    device_id: int = 0
) -> MemoryTracker:
    """
    Initializes OS/PyTorch hardware constraints and returns active tracker.
    """
    if max_ram_mb:
        apply_ram_limit_os(max_ram_mb)
    if max_vram_mb:
        apply_vram_limit(max_vram_mb, device_id)

    return MemoryTracker(max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb)
