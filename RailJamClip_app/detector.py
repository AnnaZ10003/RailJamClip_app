"""YOLO person detector skeleton.

职责：
- 加载 Ultralytics YOLO 模型
- 对单帧执行检测
- 仅输出 person 类检测框
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple


@dataclass
class Detection:
    """单个检测结果。"""

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int


class PersonDetector:
    """人物检测器（MVP）。"""

    def __init__(self, model_path: str, device: str, conf_threshold: float, iou_threshold: float, person_class_id: int = 0, max_det: int = 20) -> None:
        """初始化模型参数。

        TODO:
            - 实际加载 YOLO 模型对象。
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.max_det = max_det

    def predict_frame(self, frame_bgr: Any) -> List[Detection]:
        """对单帧做 person 检测。

        Args:
            frame_bgr: OpenCV BGR 图像。

        Returns:
            仅 person 的检测列表。

        TODO:
            - 调用 ultralytics 模型推理
            - 过滤 class_id == person
            - 组装 Detection
        """
        raise NotImplementedError
