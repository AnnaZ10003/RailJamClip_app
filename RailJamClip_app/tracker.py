"""Minimal tracker skeleton.

职责：
- 维护跨帧 track
- 关联策略固定为：IOU 优先，中心点距离兜底，否则新建 track
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from detector import Detection


@dataclass
class Track:
    """最小轨迹结构。"""

    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    last_frame_idx: int
    lost_frames: int = 0
    hit_frames: int = 0
    history: List[Tuple[int, Tuple[float, float, float, float]]] = field(default_factory=list)


class MinimalTracker:
    """MVP 追踪器。

    关联规则：
    1) 优先 IOU 关联
    2) IOU 不满足时，使用中心点距离兜底
    3) 两者都不满足时，创建新 track
    """

    def __init__(self, iou_threshold: float, max_center_distance_px: float, max_lost_frames: int, min_track_frames: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance_px = max_center_distance_px
        self.max_lost_frames = max_lost_frames
        self.min_track_frames = min_track_frames

        # TODO: 初始化 track 存储与自增 ID

    def update(self, detections: List[Detection], frame_idx: int) -> List[Track]:
        """输入当前帧检测结果，输出活跃轨迹。

        TODO:
            - 实现 IOU 优先匹配
            - 距离兜底匹配
            - unmatched detection 建新轨迹
            - 未匹配轨迹 lost_frames +1，超阈值删除
        """
        raise NotImplementedError


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """计算两个 bbox 的 IOU。

    TODO:
        - 实现标准 IOU。
    """
    raise NotImplementedError


def center_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """计算两个 bbox 中心点距离。

    TODO:
        - 实现欧氏距离。
    """
    raise NotImplementedError
