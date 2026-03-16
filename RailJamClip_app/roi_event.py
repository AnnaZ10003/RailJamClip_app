"""Three-stage ROI event logic skeleton.

职责：
- 对每个 track 维护事件状态机
- 状态顺序固定：entry -> core -> exit
- direction 仅用于 x 方向一致性过滤，不改变 ROI 语义
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from tracker import Track

Direction = Literal["left_to_right", "right_to_left"]


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


@dataclass
class EventRecord:
    """事件结果（MVP 最小字段 + 可选扩展占位）。"""

    event_id: str
    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_path: str
    hit_entry: bool
    hit_core: bool
    hit_exit: bool
    # TODO(optional): stage_frames / direction_passed / export range


class EventStateMachine:
    """轨迹级状态机。

    TODO:
        - IDLE -> SEEN_ENTRY -> SEEN_CORE -> SEEN_EXIT -> DONE
        - 连续帧阈值判定
        - 阶段超时重置
        - 方向一致性过滤
    """

    def __init__(
        self,
        entry_roi: ROI,
        core_roi: ROI,
        exit_roi: ROI,
        direction: Direction,
        min_entry_frames: int,
        min_core_frames: int,
        min_exit_frames: int,
        max_frames_between_stages: int,
        require_core: bool,
    ) -> None:
        # TODO: 保存参数并初始化 track->state 容器
        self.entry_roi = entry_roi
        self.core_roi = core_roi
        self.exit_roi = exit_roi
        self.direction = direction
        self.min_entry_frames = min_entry_frames
        self.min_core_frames = min_core_frames
        self.min_exit_frames = min_exit_frames
        self.max_frames_between_stages = max_frames_between_stages
        self.require_core = require_core

    def update_track(self, track: Track, frame_idx: int, fps: float) -> List[EventRecord]:
        """用当前 track 状态更新状态机，必要时产生事件。

        Returns:
            本帧新完成事件列表（通常 0 或 1）。
        """
        raise NotImplementedError


def point_in_roi(x: float, y: float, roi: ROI) -> bool:
    """判断点是否在 ROI 内。"""
    return roi.x <= x <= roi.x + roi.w and roi.y <= y <= roi.y + roi.h


def bbox_center(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """计算 bbox 中心点。"""
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
