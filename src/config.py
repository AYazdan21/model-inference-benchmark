import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from src.utils.logger import get_logger

logger = get_logger("ConfigLoader")

def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class TargetConfig:
    def __init__(self, targets_yaml_path: str | Path = "targets.yaml"):
        self.raw_data = load_yaml(targets_yaml_path)
        self.targets: Dict[str, Dict[str, Any]] = self.raw_data.get("targets", {})
        self.simulation: Dict[str, Any] = self.raw_data.get("simulation", {})
        self.available_targets: List[str] = list(self.targets.keys())

    def get_target(self, target_name: str) -> Dict[str, Any]:
        if target_name not in self.targets:
            raise ValueError(
                f"Unknown target '{target_name}'. Available targets: {self.available_targets}"
            )
        return self.targets[target_name]

    def device_profile(self, target_name: str):
        """Hardware profile used to simulate the target's compute speed."""
        from src.simulation.device import DeviceProfile
        return DeviceProfile.from_target(target_name, self.get_target(target_name), self.simulation)

    def list_targets(self) -> None:
        """Prints all configured targets with hardware resource limits."""
        print("\nConfigured Target Hardware Profiles & Resource Budgets:")
        print("-" * 85)
        for name, details in self.targets.items():
            desc = details.get("description", details.get("notes", "No description"))
            arch = details.get("arch", "N/A")
            acc = details.get("accelerator", "none")
            ram = f"{details.get('ram_limit_mb', 'unlimited')} MB" if details.get('ram_limit_mb') else "N/A"
            vram = f"{details.get('vram_limit_mb', 'unlimited')} MB" if details.get('vram_limit_mb') else "N/A"
            print(f" • {name:<12} | Arch: {arch:<6} | RAM: {ram:<8} | VRAM: {vram:<8} | {desc}")
        print("-" * 85 + "\n")
