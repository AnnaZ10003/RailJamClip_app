"""Minimal tracker for RailJamClip_app.

职责：
- 维护跨帧 track
- 关联策略固定为：IOU 优先，中心点距离兜底，否则新建 track
- 支持 max_lost_frames / min_track_frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Tuple

from detector import Detection


BBox = Tuple[float, float, float, float]


@dataclass
class Track:
    """最小轨迹结构。"""

    track_id: int
    bbox_xyxy: BBox
    last_frame_idx: int
    lost_frames: int = 0
    hit_frames: int = 1
    # 用于短时运动连续性：保存最近 bbox 历史
    history: List[Tuple[int, BBox]] = field(default_factory=list)


@dataclass
class TrackAssignment:
    """当前帧检测与轨迹匹配结果。"""

    detection: Detection
    track_id: int
    confirmed: bool


class MinimalTracker:
    """MVP 追踪器。

    关联规则（固定）：
    1) 优先 IOU 关联
    2) IOU 不满足时，使用中心点距离兜底
    3) 两者都不满足时，创建新 track
    """

    def __init__(self, iou_threshold: float, max_center_distance_px: float, max_lost_frames: int, min_track_frames: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance_px = max_center_distance_px
        self.max_lost_frames = max_lost_frames
        self.min_track_frames = min_track_frames

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackAssignment]:
        """输入当前帧检测结果，输出当前帧检测对应的 track 分配。"""
        # 先将已有轨迹标记为“本帧暂未匹配”
        for track in self.tracks.values():
            track.lost_frames += 1

        assignments: Dict[int, int] = {}  # det_idx -> track_id
        matched_track_ids = set()

        active_track_ids = [tid for tid, t in self.tracks.items() if t.lost_frames <= self.max_lost_frames]
        unmatched_det_idxs = list(range(len(detections)))

        # 阶段1：IOU 优先匹配（贪心）
        iou_candidates: List[Tuple[float, int, int]] = []  # (iou, tid, det_idx)
        for tid in active_track_ids:
            track = self.tracks[tid]
            for det_idx in unmatched_det_idxs:
                iou = bbox_iou(track.bbox_xyxy, detections[det_idx].bbox_xyxy)
                if iou >= self.iou_threshold:
                    iou_candidates.append((iou, tid, det_idx))

        iou_candidates.sort(key=lambda x: x[0], reverse=True)
        used_dets = set()
        for _, tid, det_idx in iou_candidates:
            if tid in matched_track_ids or det_idx in used_dets:
                continue
            self._apply_match(tid, detections[det_idx], frame_idx)
            assignments[det_idx] = tid
            matched_track_ids.add(tid)
            used_dets.add(det_idx)

        unmatched_det_idxs = [i for i in unmatched_det_idxs if i not in used_dets]

        # 阶段2：中心点距离兜底（带短时运动连续性）
        dist_candidates: List[Tuple[float, int, int]] = []  # (distance, tid, det_idx)
        candidate_track_ids = [tid for tid in active_track_ids if tid not in matched_track_ids]

        for tid in candidate_track_ids:
            track = self.tracks[tid]
            predicted_bbox = self._predict_bbox(track)
            for det_idx in unmatched_det_idxs:
                dist = center_distance(predicted_bbox, detections[det_idx].bbox_xyxy)
                if dist <= self.max_center_distance_px:
                    dist_candidates.append((dist, tid, det_idx))

        dist_candidates.sort(key=lambda x: x[0])
        used_dets = set()
        for _, tid, det_idx in dist_candidates:
            if tid in matched_track_ids or det_idx in used_dets:
                continue
            self._apply_match(tid, detections[det_idx], frame_idx)
            assignments[det_idx] = tid
            matched_track_ids.add(tid)
            used_dets.add(det_idx)

        unmatched_det_idxs = [i for i in unmatched_det_idxs if i not in used_dets]

        # 阶段3：未匹配 detection 新建轨迹
        for det_idx in unmatched_det_idxs:
            tid = self._create_track(detections[det_idx], frame_idx)
            assignments[det_idx] = tid

        # 清理丢失过久轨迹
        to_remove = [tid for tid, track in self.tracks.items() if track.lost_frames > self.max_lost_frames]
        for tid in to_remove:
            del self.tracks[tid]

        # 按输入 detection 顺序返回 track_id
        results: List[TrackAssignment] = []
        for det_idx, det in enumerate(detections):
            tid = assignments[det_idx]
            track = self.tracks.get(tid)
            confirmed = bool(track and track.hit_frames >= self.min_track_frames)
            results.append(TrackAssignment(detection=det, track_id=tid, confirmed=confirmed))
        return results

    def _create_track(self, detection: Detection, frame_idx: int) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            bbox_xyxy=detection.bbox_xyxy,
            last_frame_idx=frame_idx,
            lost_frames=0,
            hit_frames=1,
            history=[(frame_idx, detection.bbox_xyxy)],
        )
        return tid

    def _apply_match(self, track_id: int, detection: Detection, frame_idx: int) -> None:
        track = self.tracks[track_id]
        track.bbox_xyxy = detection.bbox_xyxy
        track.last_frame_idx = frame_idx
        track.lost_frames = 0
        track.hit_frames += 1
        track.history.append((frame_idx, detection.bbox_xyxy))
        # 仅保留短历史，足够支持短时运动连续性
        if len(track.history) > 5:
            track.history = track.history[-5:]

    def _predict_bbox(self, track: Track) -> BBox:
        """根据短历史做极简运动连续性预测。

        - 若历史不足 2 帧，直接用当前 bbox
        - 若历史 >=2，基于中心点速度做一步外推，再保持 bbox 尺寸
        """
        if len(track.history) < 2:
            return track.bbox_xyxy

        (_, prev_bbox), (_, last_bbox) = track.history[-2], track.history[-1]
        prev_cx, prev_cy = bbox_center(prev_bbox)
        last_cx, last_cy = bbox_center(last_bbox)

        vx = last_cx - prev_cx
        vy = last_cy - prev_cy
        pred_cx = last_cx + vx
        pred_cy = last_cy + vy

        x1, y1, x2, y2 = last_bbox
        w = x2 - x1
        h = y2 - y1
        return (pred_cx - w / 2.0, pred_cy - h / 2.0, pred_cx + w / 2.0, pred_cy + h / 2.0)


def bbox_iou(a: BBox, b: BBox) -> float:
    """计算两个 bbox 的 IOU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_center(bbox_xyxy: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(a: BBox, b: BBox) -> float:
    """计算两个 bbox 中心点欧氏距离。"""
    acx, acy = bbox_center(a)
    bcx, bcy = bbox_center(b)
    return sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)
