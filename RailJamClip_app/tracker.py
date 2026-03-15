"""Minimal tracker for RailJamClip_app.

职责：
- 维护跨帧 track
- 关联策略固定为：IOU 优先，中心点距离兜底，否则新建 track
- 支持 max_lost_frames / min_track_frames
- 支持轻量运动过滤（仅用于 confirmed 之前）
- 支持 confirmed+entered_core 轨迹的短暂保活重连
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Tuple

from detector import Detection


BBox = Tuple[float, float, float, float]
ROI = Tuple[int, int, int, int]
Direction = Literal["left_to_right", "right_to_left"]


@dataclass
class Track:
    """最小轨迹结构。"""

    track_id: int
    bbox_xyxy: BBox
    last_frame_idx: int
    lost_frames: int = 0
    hit_frames: int = 1
    confirmed: bool = False
    entered_core: bool = False
    history: List[Tuple[int, BBox]] = field(default_factory=list)


@dataclass
class TrackAssignment:
    """当前帧检测与轨迹匹配结果。"""

    detection: Detection
    track_id: int
    confirmed: bool


class MinimalTracker:
    """MVP 追踪器。"""

    def __init__(
        self,
        iou_threshold: float,
        max_center_distance_px: float,
        max_lost_frames: int,
        min_track_frames: int,
        motion_min_frames: int,
        motion_min_distance_px: float,
        direction: Direction,
        direction_min_progress_px: float,
        core_roi: ROI,
        core_reacquire_max_frames: int,
        core_reacquire_max_dist_px: float,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance_px = max_center_distance_px
        self.max_lost_frames = max_lost_frames
        self.min_track_frames = min_track_frames

        self.motion_min_frames = motion_min_frames
        self.motion_min_distance_px = motion_min_distance_px
        self.direction = direction
        self.direction_min_progress_px = direction_min_progress_px

        self.core_roi = core_roi
        self.core_reacquire_max_frames = core_reacquire_max_frames
        self.core_reacquire_max_dist_px = core_reacquire_max_dist_px

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def update(self, detections: List[Detection], frame_idx: int) -> List[TrackAssignment]:
        for track in self.tracks.values():
            track.lost_frames += 1

        assignments: Dict[int, int] = {}
        matched_track_ids = set()

        active_track_ids = [tid for tid, t in self.tracks.items() if self._within_keepalive(t)]
        unmatched_det_idxs = list(range(len(detections)))

        # 1) IOU 优先
        iou_candidates: List[Tuple[float, int, int]] = []
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

        # 2) 距离兜底（对 core 保活轨迹放宽丢失窗口但保持合理重连距离）
        dist_candidates: List[Tuple[float, int, int]] = []
        candidate_track_ids = [tid for tid in active_track_ids if tid not in matched_track_ids]
        for tid in candidate_track_ids:
            track = self.tracks[tid]
            pred_bbox = self._predict_bbox(track)
            max_dist = self.core_reacquire_max_dist_px if (track.confirmed and track.entered_core) else self.max_center_distance_px
            for det_idx in unmatched_det_idxs:
                dist = center_distance(pred_bbox, detections[det_idx].bbox_xyxy)
                if dist <= max_dist:
                    # 优先 core 保活轨迹重连：轻微减去权重
                    priority_bias = -5.0 if (track.confirmed and track.entered_core) else 0.0
                    dist_candidates.append((dist + priority_bias, tid, det_idx))

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

        # 3) 新建轨迹
        for det_idx in unmatched_det_idxs:
            tid = self._create_track(detections[det_idx], frame_idx)
            assignments[det_idx] = tid

        # 清理过久未匹配轨迹（core保活只对 confirmed+entered_core 生效）
        to_remove = [tid for tid, track in self.tracks.items() if not self._within_keepalive(track)]
        for tid in to_remove:
            del self.tracks[tid]

        results: List[TrackAssignment] = []
        for det_idx, det in enumerate(detections):
            tid = assignments[det_idx]
            track = self.tracks.get(tid)
            confirmed = bool(track and track.confirmed)
            results.append(TrackAssignment(detection=det, track_id=tid, confirmed=confirmed))
        return results

    def _within_keepalive(self, track: Track) -> bool:
        if track.confirmed and track.entered_core:
            return track.lost_frames <= self.core_reacquire_max_frames
        return track.lost_frames <= self.max_lost_frames

    def _create_track(self, detection: Detection, frame_idx: int) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            bbox_xyxy=detection.bbox_xyxy,
            last_frame_idx=frame_idx,
            lost_frames=0,
            hit_frames=1,
            confirmed=False,
            entered_core=False,
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
        if len(track.history) > 8:
            track.history = track.history[-8:]

        # 进入 core 标记（仅对已匹配轨迹更新）
        cx, cy = bbox_center(track.bbox_xyxy)
        if point_in_roi(cx, cy, self.core_roi):
            track.entered_core = True

        # 轻量运动过滤仅用于升级 confirmed 前
        if not track.confirmed:
            if track.hit_frames >= self.min_track_frames and self._motion_ok(track):
                track.confirmed = True

    def _motion_ok(self, track: Track) -> bool:
        if len(track.history) < self.motion_min_frames:
            return False
        window = track.history[-self.motion_min_frames :]
        start_cx, start_cy = bbox_center(window[0][1])
        end_cx, end_cy = bbox_center(window[-1][1])
        dx = end_cx - start_cx
        dy = end_cy - start_cy
        dist = sqrt(dx * dx + dy * dy)
        if dist < self.motion_min_distance_px:
            return False

        if self.direction == "left_to_right" and dx < self.direction_min_progress_px:
            return False
        if self.direction == "right_to_left" and dx > -self.direction_min_progress_px:
            return False
        return True

    def _predict_bbox(self, track: Track) -> BBox:
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
    acx, acy = bbox_center(a)
    bcx, bcy = bbox_center(b)
    return sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def point_in_roi(x: float, y: float, roi: ROI) -> bool:
    rx, ry, rw, rh = roi
    return rw > 0 and rh > 0 and rx <= x <= rx + rw and ry <= y <= ry + rh
