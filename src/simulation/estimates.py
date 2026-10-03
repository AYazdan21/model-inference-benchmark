from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.simulation.model_profile import profile_model, supports_dynamic_input


def estimate_matrix(model_paths: Iterable[str | Path], target_cfg) -> Dict[str, Any]:
    """
    Static model x target table of estimated on-device *inference* latency (no model execution).
    Pre/post-processing on the device CPU comes on top; run a benchmark for full-pipeline numbers.
    """
    devices = {name: target_cfg.device_profile(name) for name in target_cfg.available_targets}
    rows: List[Dict[str, Any]] = []
    for path in model_paths:
        prof = profile_model(path)
        cells = {}
        for name, dev in devices.items():
            ms = dev.estimate_inference_ms(prof)
            cells[name] = {"inference_ms": round(ms, 2), "fps": round(1000.0 / ms, 1)} if ms else None
        rows.append({**prof.to_dict(), "dynamic_input": supports_dynamic_input(path), "estimates": cells})
    return {
        "targets": {name: {"device": d.device, "runtime": d.runtime, "simulated": d.simulate} for name, d in devices.items()},
        "rows": rows,
    }


def print_matrix(matrix: Dict[str, Any]) -> None:
    targets = [t for t, info in matrix["targets"].items() if info["simulated"]]
    col = 17
    header = f"{'Model':<42}{'GFLOPs':>8}  " + "".join(f"{t:>{col}}" for t in targets)
    print("\nEstimated on-device inference (ms / FPS, model only; pre/post-processing excluded)")
    print("=" * len(header))
    print(header)
    print(f"{'':<52}" + "".join(f"{matrix['targets'][t]['device'][:col - 1]:>{col}}" for t in targets))
    print("-" * len(header))
    for row in matrix["rows"]:
        gflops = row["gflops"] if row["gflops"] is not None else "-"
        line = f"{row['name']:<42}{gflops:>8}  "
        for t in targets:
            cell = row["estimates"].get(t)
            line += f"{(str(cell['inference_ms']) + ' / ' + str(cell['fps'])) if cell else 'n/a':>{col}}"
        if row.get("error"):
            line += f"   ({row['error'][:60]})"
        print(line)
    print("=" * len(header) + "\n")
