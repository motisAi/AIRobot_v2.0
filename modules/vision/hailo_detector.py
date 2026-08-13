"""Hailo-accelerated object detector with OpenCV DNN fallback.

When the Hailo Runtime (HailoRT) is available the module loads a compiled
HEF model (e.g. ``yolov8s.hef``) and runs inference on the Hailo-8/8L AI
accelerator.  When Hailo is absent it falls back to OpenCV's DNN module
with a lightweight ONNX or caffemodel so the robot still works — just
slower.

Typical usage from other modules::

    detector = HailoDetector()
    detector.start()
    results = detector.detect(frame.image)
    for det in results:
        print(det.label, det.confidence, det.bbox)

The detector is thread-safe; multiple vision modules can call ``detect()``
concurrently.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import model_config, system_config, hardware_config

# Hailo SDK — optional
try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

# COCO class names (80 classes) used by YOLO models
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "data" / "models"


@dataclass
class Detection:
    """Single object detection result."""
    label: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float = field(default_factory=time.time)


class HailoDetector:
    """Runs YOLO inference on Hailo when available, OpenCV DNN otherwise."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.conf_threshold = model_config.object_confidence_threshold
        self.nms_threshold = model_config.object_nms_threshold
        self.max_detections = model_config.object_max_detections
        self.classes_filter: List[int] = model_config.object_classes_filter

        self._backend: str = "none"
        self._lock = threading.Lock()
        self._hailo_vdevice = None
        self._hailo_hef = None
        self._hailo_network_group = None
        self._hailo_params = None
        self._cv_net = None
        self._input_shape: Tuple[int, int] = (640, 640)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Initialize the best available backend."""
        if system_config.enable_hailo and HAILO_AVAILABLE:
            if self._init_hailo():
                return True
            self.logger.warning("Hailo init failed — falling back to OpenCV DNN")

        if self._init_opencv_dnn():
            return True

        self.logger.info(
            "Object detection off — no model present (optional camera object "
            "recognition). Add a YOLO .hef (Hailo) or .onnx to data/models/ to "
            "enable it. Face, voice, and chat are unaffected."
        )
        return False

    def stop(self) -> None:
        """Release accelerator / DNN resources."""
        with self._lock:
            if self._hailo_vdevice:
                try:
                    self._hailo_vdevice.release()
                except Exception:
                    pass
                self._hailo_vdevice = None
            self._cv_net = None
            self._backend = "none"
        self.logger.info("Detector stopped")

    @property
    def backend_name(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a BGR image and return a list of Detection."""
        if self._backend == "hailo":
            return self._detect_hailo(image)
        elif self._backend == "opencv_dnn":
            return self._detect_opencv(image)
        return []

    # ------------------------------------------------------------------
    # Hailo backend
    # ------------------------------------------------------------------
    def _init_hailo(self) -> bool:
        """Load a HEF model onto the Hailo device."""
        hef_path = MODELS_DIR / f"{model_config.object_model}.hef"
        if not hef_path.exists():
            self.logger.warning("HEF model not found at %s", hef_path)
            self.logger.info(
                "Download a compiled HEF from the Hailo Model Zoo and place it "
                "in data/models/. Example: yolov8s.hef"
            )
            return False

        try:
            self._hailo_vdevice = VDevice()
            self._hailo_hef = HEF(str(hef_path))

            configure_params = ConfigureParams.create_from_hef(
                hef=self._hailo_hef,
                interface=HailoStreamInterface.PCIe,
            )
            self._hailo_network_group = self._hailo_vdevice.configure(
                self._hailo_hef, configure_params
            )[0]

            input_vstream_info = self._hailo_hef.get_input_vstream_infos()
            if input_vstream_info:
                shape = input_vstream_info[0].shape
                # shape is typically (h, w, c)
                self._input_shape = (shape[1], shape[0])

            self._backend = "hailo"
            self.logger.info("Hailo detector ready (model=%s, input=%s)",
                             hef_path.name, self._input_shape)
            return True
        except Exception as exc:
            self.logger.error("Hailo initialization error: %s", exc)
            if self._hailo_vdevice:
                try:
                    self._hailo_vdevice.release()
                except Exception:
                    pass
                self._hailo_vdevice = None
            return False

    def _detect_hailo(self, image: np.ndarray) -> List[Detection]:
        """Run YOLO inference on the Hailo accelerator."""
        h_orig, w_orig = image.shape[:2]
        inp_w, inp_h = self._input_shape

        # Pre-process: resize + normalize
        resized = cv2.resize(image, (inp_w, inp_h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        with self._lock:
            try:
                input_vstreams_params = self._hailo_hef.create_input_vstream_params(
                    quantized=True, format_type=FormatType.UINT8
                )
                output_vstreams_params = self._hailo_hef.create_output_vstream_params(
                    quantized=False, format_type=FormatType.FLOAT32
                )

                with InferVStreams(
                    self._hailo_network_group,
                    input_vstreams_params,
                    output_vstreams_params,
                ) as pipeline:
                    input_name = list(input_vstreams_params.keys())[0]
                    raw = pipeline.infer({input_name: input_data})

                # Process raw outputs — the exact layout depends on the HEF
                # compilation.  The common Hailo Model Zoo YOLO HEFs produce
                # a dict of numpy arrays keyed by output layer name.
                return self._postprocess_yolo(raw, w_orig, h_orig, inp_w, inp_h)

            except Exception as exc:
                self.logger.error("Hailo inference error: %s", exc)
                return []

    # ------------------------------------------------------------------
    # OpenCV DNN fallback
    # ------------------------------------------------------------------
    def _init_opencv_dnn(self) -> bool:
        """Load a lightweight ONNX or Caffe model via OpenCV DNN."""
        # Try ONNX first (preferred)
        onnx_path = MODELS_DIR / f"{model_config.object_model}.onnx"
        if onnx_path.exists():
            try:
                self._cv_net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self._backend = "opencv_dnn"
                self.logger.info("OpenCV DNN detector ready (ONNX: %s)", onnx_path.name)
                return True
            except Exception as exc:
                self.logger.warning("Failed to load ONNX model: %s", exc)

        # Fallback: MobileNet SSD (Caffe)
        proto = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
        caffemodel = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
        if proto.exists() and caffemodel.exists():
            try:
                self._cv_net = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
                self._input_shape = (300, 300)
                self._backend = "opencv_dnn"
                self.logger.info("OpenCV DNN detector ready (MobileNet SSD)")
                return True
            except Exception as exc:
                self.logger.warning("Failed to load MobileNet SSD: %s", exc)

        self.logger.warning(
            "No DNN model found. Place a YOLO .onnx or MobileNet SSD .caffemodel "
            "in data/models/ for CPU fallback detection."
        )
        return False

    def _detect_opencv(self, image: np.ndarray) -> List[Detection]:
        """Run detection using the OpenCV DNN backend."""
        if self._cv_net is None:
            return []

        h_orig, w_orig = image.shape[:2]
        inp_w, inp_h = self._input_shape

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (inp_w, inp_h), swapRB=True, crop=False
        )
        self._cv_net.setInput(blob)

        try:
            outputs = self._cv_net.forward(self._cv_net.getUnconnectedOutLayersNames())
        except Exception as exc:
            self.logger.error("OpenCV DNN forward pass error: %s", exc)
            return []

        return self._postprocess_yolo_onnx(outputs, w_orig, h_orig, inp_w, inp_h)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------
    def _postprocess_yolo(self, raw_outputs: dict,
                          w_orig: int, h_orig: int,
                          inp_w: int, inp_h: int) -> List[Detection]:
        """Parse Hailo YOLO output tensors into Detection objects.

        The Hailo Model Zoo typically outputs one or more layers.  We
        concatenate and apply NMS.
        """
        detections: List[Detection] = []

        try:
            # Concatenate all output arrays
            all_outputs = []
            for key, arr in raw_outputs.items():
                if isinstance(arr, np.ndarray):
                    all_outputs.append(arr.reshape(arr.shape[0], -1))

            if not all_outputs:
                return []

            data = np.concatenate(all_outputs, axis=1).squeeze()
            if data.ndim == 1:
                return []

            # Typical YOLO output: each row = [x_center, y_center, w, h, obj_conf, cls0, cls1, ...]
            num_classes = data.shape[-1] - 5
            if num_classes <= 0:
                # Might be a different format — try transposed
                data = data.T
                num_classes = data.shape[-1] - 5
                if num_classes <= 0:
                    return []

            boxes, confidences, class_ids = [], [], []
            for row in data:
                obj_conf = row[4]
                if obj_conf < self.conf_threshold:
                    continue
                class_scores = row[5:]
                cls_id = int(np.argmax(class_scores))
                score = float(obj_conf * class_scores[cls_id])
                if score < self.conf_threshold:
                    continue
                if self.classes_filter and cls_id not in self.classes_filter:
                    continue

                cx, cy, bw, bh = row[:4]
                x = int((cx - bw / 2) * w_orig / inp_w)
                y = int((cy - bh / 2) * h_orig / inp_h)
                w = int(bw * w_orig / inp_w)
                h = int(bh * h_orig / inp_h)

                boxes.append([x, y, w, h])
                confidences.append(score)
                class_ids.append(cls_id)

            # NMS
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                           self.conf_threshold, self.nms_threshold)
                for i in indices.flatten()[:self.max_detections]:
                    label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f"class_{class_ids[i]}"
                    detections.append(Detection(
                        label=label,
                        class_id=class_ids[i],
                        confidence=confidences[i],
                        bbox=tuple(boxes[i]),
                    ))
        except Exception as exc:
            self.logger.error("YOLO postprocess error: %s", exc)

        return detections

    def _postprocess_yolo_onnx(self, outputs,
                                w_orig: int, h_orig: int,
                                inp_w: int, inp_h: int) -> List[Detection]:
        """Parse OpenCV DNN YOLO ONNX outputs."""
        detections: List[Detection] = []
        try:
            # YOLOv8 ONNX typically outputs shape (1, num_classes+4, num_detections)
            out = outputs[0]
            if out.ndim == 3:
                out = out.squeeze(0)
            if out.shape[0] < out.shape[1]:
                out = out.T  # Transpose to (num_detections, num_classes+4)

            num_classes = out.shape[1] - 4
            if num_classes <= 0:
                return []

            boxes, confidences, class_ids = [], [], []
            for row in out:
                class_scores = row[4:]
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])
                if score < self.conf_threshold:
                    continue
                if self.classes_filter and cls_id not in self.classes_filter:
                    continue

                cx, cy, bw, bh = row[:4]
                x = int((cx - bw / 2) * w_orig / inp_w)
                y = int((cy - bh / 2) * h_orig / inp_h)
                w = int(bw * w_orig / inp_w)
                h = int(bh * h_orig / inp_h)

                boxes.append([x, y, w, h])
                confidences.append(score)
                class_ids.append(cls_id)

            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                           self.conf_threshold, self.nms_threshold)
                for i in indices.flatten()[:self.max_detections]:
                    label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f"class_{class_ids[i]}"
                    detections.append(Detection(
                        label=label,
                        class_id=class_ids[i],
                        confidence=confidences[i],
                        bbox=tuple(boxes[i]),
                    ))
        except Exception as exc:
            self.logger.error("ONNX YOLO postprocess error: %s", exc)

        return detections
