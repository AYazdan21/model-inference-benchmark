import ast
import time
from typing import Optional
import numpy as np
import cv2
from src.runtimes.base import BaseDetector, Detection, DetectionResult
from src.utils.logger import get_logger

logger = get_logger("ONNXRunner")

def classify_output(raw_output: np.ndarray) -> str:
    """Recognises the output heads the runner can decode into boxes."""
    if raw_output.ndim == 3 and raw_output.shape[2] == 6 and raw_output.shape[1] > 6:
        return "end2end"  # NMS-free (YOLOv10/YOLO26-style) head: [1, max_det, 6]
    if raw_output.ndim == 3 and raw_output.shape[1] > 4 and raw_output.shape[1] < raw_output.shape[2]:
        return "yolo"     # YOLOv8/11-style head: [1, 4 + classes, anchors]
    return "unsupported"

def is_retinaface(outputs) -> bool:
    """RetinaFace-style graph: three outputs [1, N, 4] box offsets, [1, N, 2] bg/face scores, [1, N, 10] landmarks."""
    return len(outputs) == 3 and [o.shape[-1] for o in outputs] == [4, 2, 10]

def retinaface_priors(height: int, width: int) -> np.ndarray:
    """Anchor centres and sizes (normalised cx, cy, w, h) of the standard RetinaFace configuration."""
    min_sizes, steps = ((16, 32), (64, 128), (256, 512)), (8, 16, 32)
    priors = []
    for sizes, step in zip(min_sizes, steps):
        fh, fw = int(np.ceil(height / step)), int(np.ceil(width / step))
        cy, cx = np.meshgrid((np.arange(fh) + 0.5) * step / height, (np.arange(fw) + 0.5) * step / width, indexing="ij")
        for size in sizes:  # anchors interleave per cell: (cell 0, size 0), (cell 0, size 1), ...
            priors.append(np.stack([cx, cy, np.full_like(cx, size / width), np.full_like(cy, size / height)], -1))
    return np.concatenate([np.stack(priors[i:i + 2], 2).reshape(-1, 4) for i in range(0, len(priors), 2)]).astype(np.float32)

class ONNXDetector(BaseDetector):
    def __init__(self, model_path: str, config: Optional[dict] = None):
        super().__init__(model_path, config)
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = {}
        # How the output head was decoded: "yolo", "end2end", "retinaface", or "unsupported" (timed only, no boxes)
        self.output_format: Optional[str] = None
        self._priors: dict = {}  # RetinaFace anchors per input (height, width)
        self.conf = self.config.get("model", {}).get("conf_threshold", 0.35)
        self.iou = self.config.get("model", {}).get("iou_threshold", 0.45)
        self.input_size = tuple(self.config.get("model", {}).get("input_size", [640, 640]))
        self.dynamic_input = False  # True when the graph has symbolic / dynamic height and width (set by load_model)
        self.load_model()

    def load_model(self) -> None:
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            # Check if CUDA is requested/available
            device = self.config.get("inference", {}).get("device", "cpu")
            if device.lower() in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            opts = ort.SessionOptions()
            num_threads = self.config.get("inference", {}).get("num_threads")
            if num_threads:
                opts.intra_op_num_threads = int(num_threads)

            logger.info(f"Loading ONNX model: {self.model_path} with providers {providers} (threads: {num_threads or 'auto'})")
            self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.output_names = [o.name for o in self.session.get_outputs()]
            if is_retinaface(self.session.get_outputs()):
                self.output_format = "retinaface"
                self.class_names = {0: "face"}
                logger.info("RetinaFace-style face detector (box offsets, scores, landmarks)")

            # Ultralytics exports store class names in the model metadata
            names_meta = self.session.get_modelmeta().custom_metadata_map.get("names")
            if names_meta:
                try:
                    self.class_names = {int(k): str(v).strip() for k, v in ast.literal_eval(names_meta).items()}
                except (ValueError, SyntaxError):
                    pass

            # Dynamically detect input size from model graph if specified
            if len(inp.shape) == 4:
                h_dim, w_dim = inp.shape[2], inp.shape[3]
                if isinstance(h_dim, int) and isinstance(w_dim, int) and h_dim > 0 and w_dim > 0:
                    self.input_size = (h_dim, w_dim)
                    logger.info(f"Detected model input resolution: {self.input_size}")
                else:
                    self.dynamic_input = True
                    logger.info(f"Model accepts a dynamic input size (default {self.input_size})")
        except ImportError:
            raise ImportError("onnxruntime is required for ONNXDetector. Run: pip install onnxruntime")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        if self.output_format == "retinaface":
            # RetinaFace is trained on BGR pixels minus the channel means (no 0-1 scaling)
            tensor = canvas.astype(np.float32) - np.array([104.0, 117.0, 123.0], dtype=np.float32)
        else:
            tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR (OpenCV) -> RGB (model)
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC to CHW
        tensor = np.expand_dims(tensor, axis=0)   # BCHW
        return tensor, scale, (left, top)

    def _decode_retinaface(self, outputs, input_hw, scale: float, pad_x: int, pad_y: int) -> list:
        loc, conf = outputs[0][0], outputs[1][0]
        h, w = int(input_hw[0]), int(input_hw[1])
        if (h, w) not in self._priors:
            self._priors[(h, w)] = retinaface_priors(h, w)
        priors = self._priors[(h, w)]

        scores = conf[:, 1]
        keep = scores >= self.conf
        loc, priors, scores = loc[keep], priors[keep], scores[keep]
        if len(scores) == 0:
            return []
        # Decode offsets with the standard variances (0.1 centre, 0.2 size) into input pixels, then undo the letterbox
        centres = priors[:, :2] + loc[:, :2] * 0.1 * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        boxes = np.concatenate([centres - sizes / 2, centres + sizes / 2], 1) * [w, h, w, h]
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / scale

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
            scores=[float(s) for s in scores],
            score_threshold=self.conf,
            nms_threshold=self.iou
        )
        return [Detection(box=[float(v) for v in boxes[i]], confidence=float(scores[i]), class_id=0, class_name="face")
                for i in np.array(indices).flatten()]

    def predict(self, frame: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        
        # Preprocessing
        t_pre0 = time.perf_counter()
        input_tensor, scale, (pad_x, pad_y) = self._preprocess(frame)
        t_pre1 = time.perf_counter()

        # Inference
        t_inf0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        t_inf1 = time.perf_counter()

        # Postprocessing
        t_post0 = time.perf_counter()
        detections = []
        raw_output = outputs[0]
        if self.output_format == "retinaface":
            detections = self._decode_retinaface(outputs, input_tensor.shape[2:], scale, pad_x, pad_y)
            t_post1 = time.perf_counter()
            return DetectionResult(
                detections=detections,
                latency_ms=(t_post1 - t0) * 1000.0,
                preprocess_ms=(t_pre1 - t_pre0) * 1000.0,
                inference_ms=(t_inf1 - t_inf0) * 1000.0,
                postprocess_ms=(t_post1 - t_post0) * 1000.0
            )
        self.output_format = classify_output(raw_output)

        # End-to-end (NMS-free, YOLOv10/YOLO26-style) head: [1, max_det, 6] = x1, y1, x2, y2, score, class
        if self.output_format == "end2end":
            preds = raw_output[0]
            preds = preds[preds[:, 4] >= self.conf]
            for x1, y1, x2, y2, score, cls in preds:
                cls_id = int(cls)
                detections.append(Detection(
                    box=[float((x1 - pad_x) / scale), float((y1 - pad_y) / scale),
                         float((x2 - pad_x) / scale), float((y2 - pad_y) / scale)],
                    confidence=float(score),
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, str(cls_id))
                ))
            t_post1 = time.perf_counter()
            return DetectionResult(
                detections=detections,
                latency_ms=(t_post1 - t0) * 1000.0,
                preprocess_ms=(t_pre1 - t_pre0) * 1000.0,
                inference_ms=(t_inf1 - t_inf0) * 1000.0,
                postprocess_ms=(t_post1 - t_post0) * 1000.0
            )

        # Otherwise only YOLOv8/11-style heads ([1, 4 + classes, anchors]) are decoded; other models
        # (classifiers, embeddings, face/pose heads) are timed but produce no boxes
        if self.output_format == "unsupported":
            t_post1 = time.perf_counter()
            return DetectionResult(
                detections=[],
                latency_ms=(t_post1 - t0) * 1000.0,
                preprocess_ms=(t_pre1 - t_pre0) * 1000.0,
                inference_ms=(t_inf1 - t_inf0) * 1000.0,
                postprocess_ms=(t_post1 - t_post0) * 1000.0
            )

        preds = np.transpose(raw_output[0], (1, 0))  # [num_boxes, 4 + classes]
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) > 0:
            # Convert xywh to xyxy and unpad
            x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
            y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
            x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale
            y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale

            # NMS
            indices = cv2.dnn.NMSBoxes(
                bboxes=[[float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i])] for i in range(len(boxes))],
                scores=[float(c) for c in confidences],
                score_threshold=self.conf,
                nms_threshold=self.iou
            )

            if len(indices) > 0:
                for idx in indices.flatten():
                    detections.append(Detection(
                        box=[float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                        confidence=float(confidences[idx]),
                        class_id=int(class_ids[idx]),
                        class_name=self.class_names.get(int(class_ids[idx]), str(class_ids[idx]))
                    ))
        t_post1 = time.perf_counter()

        t_end = time.perf_counter()
        return DetectionResult(
            detections=detections,
            latency_ms=(t_end - t0) * 1000.0,
            preprocess_ms=(t_pre1 - t_pre0) * 1000.0,
            inference_ms=(t_inf1 - t_inf0) * 1000.0,
            postprocess_ms=(t_post1 - t_post0) * 1000.0
        )
