"""YOLO person detector for RailJamClip_app.

职责：
- 加载 Ultralytics YOLO 模型
- 对单帧执行检测
- 仅输出 person 类检测框
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Detection:
    """单个检测结果。"""

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int


class PersonDetector:
    """人物检测器（MVP）。"""

    def __init__(
        self,
        model_path: str,
        device: str,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int = 640,
        person_class_id: int = 0,
        max_det: int = 20,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.person_class_id = person_class_id
        self.max_det = max_det

        # 延迟加载，避免未使用检测链路时强依赖。
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def predict_frame(self, frame_bgr: Any) -> List[Detection]:
        """对单帧做 person 检测并返回 person 框。"""
        # CPU 可用性优化：
        # 1) 使用配置的 imgsz 控制推理输入尺寸
        # 2) 直接在模型层过滤 classes=[person_class_id]，减少后处理开销
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.person_class_id],
            max_det=self.max_det,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            detections.append(
                Detection(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    class_id=cls_id,
                )
            )
        return detections
