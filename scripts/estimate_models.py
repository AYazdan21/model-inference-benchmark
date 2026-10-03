"""
Instant model x target comparison: estimated on-device inference latency for every model,
derived from model GFLOPs and the calibration anchors in targets.yaml. Nothing is executed.

    python scripts/estimate_models.py                    # all models in models/
    python scripts/estimate_models.py --model models/Gun_Detection_input_640.onnx --json out.json
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TargetConfig
from src.simulation.estimates import estimate_matrix, print_matrix

MODEL_EXTS = {".onnx", ".pt"}


def main():
    parser = argparse.ArgumentParser(description="Estimate model latency on every target in targets.yaml")
    parser.add_argument("--model", nargs="*", default=None, help="Model paths (default: all .onnx/.pt in models/)")
    parser.add_argument("--json", default=None, help="Also write the table as JSON to this path")
    args = parser.parse_args()

    models = args.model or sorted(str(p) for p in (PROJECT_ROOT / "models").glob("*") if p.suffix.lower() in MODEL_EXTS)
    matrix = estimate_matrix(models, TargetConfig(PROJECT_ROOT / "targets.yaml"))
    print_matrix(matrix)
    if args.json:
        Path(args.json).write_text(json.dumps(matrix, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
