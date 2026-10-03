"""
Static model profiling: compute cost (GFLOPs) and parameter count.

GFLOPs follow the Ultralytics / Hailo convention (2 x multiply-accumulates),
so YOLOv8n @ 640 ~= 8.7 GFLOPs. Only Conv / ConvTranspose / Gemm / MatMul are
counted; element-wise ops are negligible for CNN detectors.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from src.utils.logger import get_logger

logger = get_logger("ModelProfile")


@dataclass
class ModelProfile:
    name: str
    gflops: Optional[float]      # None if it could not be determined
    params_m: Optional[float]
    input_size: Tuple[int, int]  # (height, width) the GFLOPs were computed at
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def profile_model(model_path: str | Path, input_size: Tuple[int, int] = (640, 640)) -> ModelProfile:
    path = Path(model_path)
    return _profile_cached(str(path.resolve()), path.stat().st_mtime if path.exists() else 0.0, tuple(input_size))


def supports_dynamic_input(model_path: str | Path) -> Optional[bool]:
    """True when the model accepts input sizes other than its default: Ultralytics .pt weights, and ONNX graphs whose
    height and width are symbolic. False for a static input (ONNX Runtime rejects other sizes); None if unknown."""
    path = Path(model_path)
    ext = path.suffix.lower()
    if ext in (".pt", ".pth"):
        return True
    if ext != ".onnx":
        return None
    try:
        return _onnx_dynamic_cached(str(path.resolve()), path.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read the input shape of {path.name}: {e}")
        return None


@lru_cache(maxsize=64)
def _onnx_dynamic_cached(path_str: str, _mtime: float) -> bool:
    import onnx
    model = onnx.load(path_str, load_external_data=False)
    init_names = {t.name for t in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        dims = inp.type.tensor_type.shape.dim
        if len(dims) != 4:
            return False
        return not all(d.dim_value > 0 for d in dims[2:])
    return False


@lru_cache(maxsize=64)
def _profile_cached(path_str: str, _mtime: float, input_size: Tuple[int, int]) -> ModelProfile:
    path = Path(path_str)
    ext = path.suffix.lower()
    try:
        if ext == ".onnx":
            return _profile_onnx(path, input_size)
        if ext in (".pt", ".pth"):
            return _profile_torch(path, input_size)
        return ModelProfile(path.name, None, None, input_size, error=f"Cannot profile '{ext}' files")
    except Exception as e:
        logger.warning(f"Could not profile {path.name}: {e}")
        return ModelProfile(path.name, None, None, input_size, error=str(e))


def _profile_onnx(path: Path, default_size: Tuple[int, int]) -> ModelProfile:
    import onnx
    from onnx import shape_inference

    # Weights are not needed for counting, and some models keep them in external files
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    init_names = {t.name for t in graph.initializer}
    params = sum(int(np.prod(t.dims)) for t in graph.initializer)

    # Pin dynamic dims: batch -> 1, spatial -> default_size
    inputs = [i for i in graph.input if i.name not in init_names]
    input_hw = default_size
    for inp in inputs:
        dims = inp.type.tensor_type.shape.dim
        for idx, d in enumerate(dims):
            if d.dim_value > 0:
                continue
            if idx == 0:
                d.dim_value = 1
            elif len(dims) == 4 and idx in (2, 3):
                d.dim_value = default_size[idx - 2]
            else:
                d.dim_value = 1
        if len(dims) == 4:
            input_hw = (dims[2].dim_value, dims[3].dim_value)

    # Exported value_info can carry stale symbolic dims (e.g. 'batch'); re-infer from the pinned inputs
    del graph.value_info[:]
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(x > 0 for x in dims):
            shapes[vi.name] = dims
    for t in graph.initializer:
        shapes[t.name] = list(t.dims)

    compute_ops = [n for n in inferred.graph.node if n.op_type in ("Conv", "ConvTranspose", "Gemm", "MatMul")]
    missing = {name for n in compute_ops for name in n.input[:2] + n.output[:1] if name and name not in shapes}
    if missing:
        shapes.update(_runtime_shapes(model, sorted(missing), inputs))

    macs = 0
    uncounted = 0
    for node in compute_ops:
        out = shapes.get(node.output[0])
        a = shapes.get(node.input[0])
        b = shapes.get(node.input[1]) if len(node.input) > 1 else None
        if out is None or b is None:
            uncounted += 1
            continue
        if node.op_type == "Conv":
            # weight: [Cout, Cin/groups, kH, kW]
            macs += int(np.prod(out)) * int(np.prod(b[1:]))
        elif node.op_type == "ConvTranspose":
            # weight: [Cin, Cout/groups, kH, kW]; cost scales with input size
            if a is None:
                uncounted += 1
                continue
            macs += int(np.prod(a)) * int(np.prod(b[1:]))
        elif node.op_type == "Gemm":
            trans_b = any(at.name == "transB" and at.i == 1 for at in node.attribute)
            k = b[1] if trans_b else b[0]
            macs += int(np.prod(out)) * k
        else:  # MatMul
            macs += int(np.prod(out)) * (b[-2] if len(b) >= 2 else b[0])

    if uncounted:
        logger.warning(f"{path.name}: {uncounted} compute ops had unknown shapes and were not counted")

    return ModelProfile(path.name, round(2 * macs / 1e9, 3), round(params / 1e6, 3), tuple(input_hw))


def _runtime_shapes(model, tensor_names, inputs) -> dict:
    """Fallback when static shape inference fails: run once in ONNX Runtime and read the real shapes."""
    try:
        import onnx
        import onnxruntime as ort
        from onnx import helper, TensorProto

        probe = onnx.ModelProto()
        probe.CopyFrom(model)
        existing = {o.name for o in probe.graph.output}
        for name in tensor_names:
            if name not in existing:
                probe.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(probe.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        feed = {}
        for inp in inputs:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            dtype = helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
            feed[inp.name] = np.zeros(dims, dtype=dtype)
        outs = sess.run(tensor_names, feed)
        return {name: list(arr.shape) for name, arr in zip(tensor_names, outs)}
    except Exception as e:
        logger.debug(f"Runtime shape probe failed: {e}")
        return {}


def _profile_torch(path: Path, input_size: Tuple[int, int]) -> ModelProfile:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    yolo = YOLO(str(path))
    net = yolo.model
    params = sum(p.numel() for p in net.parameters())
    gflops = get_flops(net, imgsz=list(input_size))
    return ModelProfile(path.name, round(float(gflops), 3) if gflops else None, round(params / 1e6, 3), tuple(input_size))
