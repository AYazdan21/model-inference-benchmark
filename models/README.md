# Models Directory

Place your trained detection model weights in this directory.

## Target-Specific Format Reference

Depending on the target hardware specified in `targets.yaml`, models should be converted into their respective optimized runtime formats:

| Target ID | Hardware Platform | Expected Weight Format | Notes / Tooling |
|---|---|---|---|
| `x86-cpu` | Intel / AMD x86_64 CPU | `.pt`, `.onnx`, OpenVINO | Native PyTorch or ONNX Runtime CPU |
| `arm64-cpu` | ARM Cortex-A (RPi, RK3588, etc.) | `.onnx`, `.pt` | ONNX Runtime ARM64 build |
| `jetson` | Nvidia Jetson (Nano, Orin, Xavier) | `.engine`, `.onnx` | TensorRT optimized engine via `trtexec` |
| `rk3588-npu` | Rockchip RK3588 NPU | `.rknn` | Converted via `rknn-toolkit2` |
| `hailo10h` | Hailo-10H M.2 AI Acceleration Module | `.hef` | Compiled via Hailo Dataflow Compiler (DFC) |

## Quick Export Examples
- Export YOLO to ONNX:
  ```bash
  yolo export model=models/best-yolo11-seg.pt format=onnx imgsz=640
  ```
- Export ONNX to TensorRT Engine (Jetson):
  ```bash
  trtexec --onnx=models/model.onnx --saveEngine=models/model.engine --fp16
  ```
