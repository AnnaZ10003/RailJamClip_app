"""Main entry for RailJamClip_app MVP.

当前实现范围（本轮）：
- 读取 config.yaml
- 校验输入视频路径
- 读取视频基础信息
- 创建 output/clips 与 logs 目录
- 输出首帧 calibration 标定图（ROI 可视化）
- 接入 YOLO person 检测并统计检测结果
- 接入最小 tracking（IOU 优先 + 中心点距离兜底 + 轻量保活）
- active_frame_roi 约束所有 ROI/tracking 绘制与过滤
- 可选输出每帧检测+track 调试 JSON
- 可选输出 tracking 预览视频（bbox/track_id/confirmed + ROI）
- 生成并写出 metadata.json（当前不做 roi_event/clips）
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from detector import Detection, PersonDetector
from metadata import build_metadata, write_metadata
from tracker import MinimalTracker, TrackAssignment
from utils import ensure_dir, load_config, read_video_info, write_json

BBox = Tuple[float, float, float, float]
ROI = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RailJamClip_app MVP runner")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config yaml (default: config.yaml)")
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _detect_repo_name_fallback() -> str:
    return Path(__file__).resolve().parent.name


def _build_run_info(config: Dict[str, Any], started_at: str, finished_at: str, status: str) -> Dict[str, Any]:
    project_name = config.get("project", {}).get("project_name") or _detect_repo_name_fallback()
    run_name = config.get("project", {}).get("run_name", "default_run")
    return {
        "project_name": project_name,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
    }


def _to_debug_item(frame_idx: int, assignments: List[TrackAssignment]) -> Dict[str, Any]:
    return {
        "frame_index": frame_idx,
        "person_count": len(assignments),
        "detections": [
            {
                "track_id": a.track_id,
                "bbox_xyxy": [a.detection.bbox_xyxy[0], a.detection.bbox_xyxy[1], a.detection.bbox_xyxy[2], a.detection.bbox_xyxy[3]],
                "confidence": a.detection.confidence,
                "class_id": a.detection.class_id,
                "confirmed": a.confirmed,
            }
            for a in assignments
        ],
    }


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_roi(x: float, y: float, roi: ROI) -> bool:
    rx, ry, rw, rh = roi
    return rw > 0 and rh > 0 and rx <= x <= rx + rw and ry <= y <= ry + rh


def _parse_roi(roi_cfg: Dict[str, Any], key: str) -> ROI:
    roi = roi_cfg.get(key, {})
    return int(roi.get("x", 0)), int(roi.get("y", 0)), int(roi.get("w", 0)), int(roi.get("h", 0))


def _clip_roi_to_bounds(roi: ROI, width: int, height: int) -> Tuple[ROI, bool]:
    x, y, w, h = roi
    x1 = max(0, min(width - 1, x)) if width > 0 else 0
    y1 = max(0, min(height - 1, y)) if height > 0 else 0
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    clipped = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
    return clipped, clipped != roi


def _clip_roi_to_parent(roi: ROI, parent: ROI) -> Tuple[ROI, bool]:
    x, y, w, h = roi
    px, py, pw, ph = parent
    px2, py2 = px + pw, py + ph
    x1 = max(px, min(px2, x))
    y1 = max(py, min(py2, y))
    x2 = max(px, min(px2, x + w))
    y2 = max(py, min(py2, y + h))
    clipped = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
    return clipped, clipped != roi


def _roi_to_cfg(roi: ROI) -> Dict[str, int]:
    x, y, w, h = roi
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _warn_if_small_roi(name: str, roi: ROI, frame_w: int, frame_h: int, min_w: int, min_h: int, min_area_ratio: float) -> None:
    _, _, rw, rh = roi
    area = rw * rh
    area_threshold = frame_w * frame_h * min_area_ratio
    if rw < min_w or rh < min_h or area < area_threshold:
        print(f"[WARN] ROI '{name}' may be too small after clipping: w={rw}, h={rh}, area={area}, area_threshold={int(area_threshold)}")


def _build_tracking_roi(stage_rois: List[ROI], margin_px: int, active_frame_roi: ROI) -> ROI:
    valid = [r for r in stage_rois if r[2] > 0 and r[3] > 0]
    if not valid:
        return (0, 0, 0, 0)

    min_x = min(r[0] for r in valid)
    min_y = min(r[1] for r in valid)
    max_x = max(r[0] + r[2] for r in valid)
    max_y = max(r[1] + r[3] for r in valid)

    roi = (min_x - margin_px, min_y - margin_px, (max_x - min_x) + 2 * margin_px, (max_y - min_y) + 2 * margin_px)
    clipped, changed = _clip_roi_to_parent(roi, active_frame_roi)
    if changed:
        print(f"[WARN] tracking_roi auto-generated and clipped from {roi} to {clipped}")
    return clipped


def _draw_roi(frame_bgr: Any, roi: ROI, label: str, color: Tuple[int, int, int], with_coords: bool = False) -> None:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    title = label if not with_coords else f"{label} ({x},{y},{w},{h})"
    cv2.putText(frame_bgr, title, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_tracking_overlay(frame_bgr: Any, frame_idx: int, assignments: List[TrackAssignment], clipped_roi_cfg: Dict[str, Any]) -> Any:
    _draw_roi(frame_bgr, _parse_roi(clipped_roi_cfg, "active_frame_roi"), "active_frame_roi", (200, 200, 200))
    _draw_roi(frame_bgr, _parse_roi(clipped_roi_cfg, "tracking_roi"), "tracking_roi", (180, 0, 180))
    _draw_roi(frame_bgr, _parse_roi(clipped_roi_cfg, "entry_roi"), "entry_roi", (255, 200, 0))
    _draw_roi(frame_bgr, _parse_roi(clipped_roi_cfg, "core_roi"), "core_roi", (255, 255, 0))
    _draw_roi(frame_bgr, _parse_roi(clipped_roi_cfg, "exit_roi"), "exit_roi", (0, 200, 255))

    for a in assignments:
        x1, y1, x2, y2 = map(int, a.detection.bbox_xyxy)
        color = (0, 220, 0) if a.confirmed else (0, 80, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        tag = f"id={a.track_id} conf={'T' if a.confirmed else 'F'} p={a.detection.confidence:.2f}"
        cv2.putText(frame_bgr, tag, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    info = f"frame={frame_idx} persons={len(assignments)}"
    cv2.putText(frame_bgr, info, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr


def _render_calibration_preview(first_frame: Any, clipped_roi_cfg: Dict[str, Any], out_path: Path) -> None:
    frame = first_frame.copy()
    fh, fw = frame.shape[:2]

    # 整帧边界
    cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (255, 255, 255), 2)
    cv2.putText(frame, f"frame (0,0,{fw},{fh})", (12, fh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    active = _parse_roi(clipped_roi_cfg, "active_frame_roi")
    entry = _parse_roi(clipped_roi_cfg, "entry_roi")
    core = _parse_roi(clipped_roi_cfg, "core_roi")
    exit_roi = _parse_roi(clipped_roi_cfg, "exit_roi")

    _draw_roi(frame, active, "active_frame_roi", (210, 210, 210), with_coords=True)
    _draw_roi(frame, entry, "entry_roi", (255, 200, 0), with_coords=True)
    _draw_roi(frame, core, "core_roi", (255, 255, 0), with_coords=True)
    _draw_roi(frame, exit_roi, "exit_roi", (0, 200, 255), with_coords=True)

    # 左上角再次明确 active 参数
    ax, ay, aw, ah = active
    cv2.putText(frame, f"active_frame_roi: x={ax}, y={ay}, w={aw}, h={ah}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"[INFO] Calibration preview saved: {out_path}")


def _filter_detections_for_tracking(
    detections: List[Detection],
    tracking_cfg: Dict[str, Any],
    active_frame_roi: ROI,
    tracking_roi: ROI,
) -> List[Detection]:
    use_tracking_roi = bool(tracking_cfg.get("use_tracking_roi", True))
    min_w = float(tracking_cfg.get("min_bbox_width_px", 18))
    min_h = float(tracking_cfg.get("min_bbox_height_px", 24))
    min_area = float(tracking_cfg.get("min_bbox_area_px", 500))

    filtered: List[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        if w < min_w or h < min_h or area < min_area:
            continue

        cx, cy = _bbox_center(d.bbox_xyxy)
        if not _point_in_roi(cx, cy, active_frame_roi):
            continue

        if use_tracking_roi and tracking_roi[2] > 0 and tracking_roi[3] > 0 and not _point_in_roi(cx, cy, tracking_roi):
            continue

        filtered.append(d)
    return filtered


def run_pipeline(config_path: Path) -> int:
    started_at = _utc_now_iso()
    config = load_config(config_path)
    print(f"[INFO] 已读取配置: {config_path}")

    input_video_path = Path(config["input"]["video_path"])
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    video_info = read_video_info(input_video_path)
    frame_w, frame_h = int(video_info["width"]), int(video_info["height"])
    print(f"[INFO] 已打开视频: {input_video_path}")
    print(
        "[INFO] 视频基础信息: "
        f"fps={video_info['fps']}, total_frames={video_info['total_frames']}, width={frame_w}, height={frame_h}, "
        f"duration_seconds={video_info['duration_seconds']:.3f}"
    )

    clips_dir = Path(config["output"]["clips_dir"])
    debug_log_path = Path(config["output"]["debug_log_path"])
    metadata_path = Path(config["output"]["metadata_path"])

    debug_cfg = config.get("debug", {})
    debug_json_enabled = bool(debug_cfg.get("export_detection_json", True))
    debug_json_path = Path(debug_cfg.get("detection_json_path", "output/detections_debug.json"))
    preview_enabled = bool(debug_cfg.get("export_preview_video", True))
    preview_path = Path(debug_cfg.get("preview_video_path", "output/preview_tracking.mp4"))

    calibration_cfg = config.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("export_calibration_preview", True))
    calibration_preview_path = Path(calibration_cfg.get("preview_image_path", "output/calibration_preview.jpg"))

    ensure_dir(clips_dir)
    ensure_dir(debug_log_path.parent)

    roi_cfg = config.get("roi", {})
    tracking_cfg = config.get("tracking", {})

    roi_min_w = int(tracking_cfg.get("min_roi_width_px", 24))
    roi_min_h = int(tracking_cfg.get("min_roi_height_px", 24))
    roi_min_area_ratio = float(tracking_cfg.get("min_roi_area_ratio", 0.001))

    raw_active = _parse_roi(roi_cfg, "active_frame_roi")
    active_frame_roi, active_changed = _clip_roi_to_bounds(raw_active, frame_w, frame_h)
    if active_changed:
        print(f"[WARN] ROI 'active_frame_roi' clipped from {raw_active} to {active_frame_roi}")
    _warn_if_small_roi("active_frame_roi", active_frame_roi, frame_w, frame_h, roi_min_w, roi_min_h, roi_min_area_ratio)

    clipped_roi_cfg: Dict[str, Any] = {"active_frame_roi": _roi_to_cfg(active_frame_roi)}

    stage_rois: List[ROI] = []
    for key in ["entry_roi", "core_roi", "exit_roi"]:
        raw_stage = _parse_roi(roi_cfg, key)
        clipped_stage, changed = _clip_roi_to_parent(raw_stage, active_frame_roi)
        if changed:
            print(f"[WARN] ROI '{key}' clipped to active_frame_roi from {raw_stage} to {clipped_stage}")
        _warn_if_small_roi(key, clipped_stage, frame_w, frame_h, roi_min_w, roi_min_h, roi_min_area_ratio)
        clipped_roi_cfg[key] = _roi_to_cfg(clipped_stage)
        stage_rois.append(clipped_stage)

    tracking_roi = _build_tracking_roi(stage_rois, int(tracking_cfg.get("tracking_roi_margin_px", 50)), active_frame_roi)
    _warn_if_small_roi("tracking_roi", tracking_roi, frame_w, frame_h, roi_min_w, roi_min_h, roi_min_area_ratio)
    clipped_roi_cfg["tracking_roi"] = _roi_to_cfg(tracking_roi)

    # 先取首帧用于标定图
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for detection: {input_video_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read first frame for calibration preview")

    if calibration_enabled:
        _render_calibration_preview(first_frame, clipped_roi_cfg, calibration_preview_path)

    detector_cfg = config.get("detector", {})
    frame_step = int(detector_cfg.get("frame_step", 2))
    if frame_step <= 0:
        cap.release()
        raise ValueError("detector.frame_step must be >= 1")
    print(f"[INFO] 当前抽帧步长 detector.frame_step={frame_step}")

    print("[INFO] 正在加载 YOLO 模型...")
    detector = PersonDetector(
        model_path=detector_cfg["model_path"],
        device=detector_cfg.get("device", "cpu"),
        conf_threshold=float(detector_cfg.get("conf_threshold", 0.35)),
        iou_threshold=float(detector_cfg.get("iou_threshold", 0.5)),
        imgsz=int(detector_cfg.get("imgsz", 640)),
        person_class_id=int(detector_cfg.get("person_class_id", 0)),
        max_det=int(detector_cfg.get("max_det", 20)),
    )
    print("[INFO] YOLO 模型已加载")

    core_roi = _parse_roi(clipped_roi_cfg, "core_roi")
    tracker = MinimalTracker(
        iou_threshold=float(tracking_cfg.get("iou_threshold", 0.25)),
        max_center_distance_px=float(tracking_cfg.get("max_center_distance_px", 80)),
        max_lost_frames=int(tracking_cfg.get("max_lost_frames", 8)),
        min_track_frames=int(tracking_cfg.get("min_track_frames", 6)),
        motion_min_frames=int(tracking_cfg.get("min_motion_frames", 3)),
        motion_min_distance_px=float(tracking_cfg.get("min_motion_distance_px", 10)),
        direction=str(config.get("event", {}).get("direction", "left_to_right")),
        direction_min_progress_px=float(tracking_cfg.get("direction_min_progress_px", 5)),
        core_roi=core_roi,
        core_reacquire_max_frames=int(tracking_cfg.get("core_reacquire_max_frames", 10)),
        core_reacquire_max_dist_px=float(tracking_cfg.get("core_reacquire_max_dist_px", 120)),
    )

    writer = None
    if preview_enabled:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        base_fps = float(video_info["fps"])
        out_fps = max(1.0, base_fps / frame_step) if base_fps > 0 else 10.0
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (frame_w, frame_h))

    total_frames = int(video_info.get("total_frames", 0))
    processed_frames = 0
    frames_with_person = 0
    total_person_boxes = 0
    debug_frames: List[Dict[str, Any]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        detections = detector.predict_frame(frame)
        filtered = _filter_detections_for_tracking(detections, tracking_cfg, active_frame_roi, tracking_roi)
        assignments = tracker.update(filtered, frame_idx)

        processed_frames += 1
        total_person_boxes += len(assignments)
        if assignments:
            frames_with_person += 1

        if debug_json_enabled:
            debug_frames.append(_to_debug_item(frame_idx, assignments))

        if writer is not None:
            preview_frame = _draw_tracking_overlay(frame.copy(), frame_idx, assignments, clipped_roi_cfg)
            writer.write(preview_frame)

        if processed_frames % 50 == 0:
            print(f"[INFO] 检测进度: {frame_idx + 1}/{total_frames}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] 预览视频已输出: {preview_path}")

    if debug_json_enabled:
        write_json(
            {
                "video_path": str(input_video_path),
                "frame_step": frame_step,
                "processed_frames": processed_frames,
                "frames_with_person": frames_with_person,
                "total_person_boxes": total_person_boxes,
                "frames": debug_frames,
            },
            debug_json_path,
        )

    events: List[Dict[str, Any]] = []
    summary = {"candidate_events": 0, "qualified_events": 0, "exported_clips": 0, "dropped_events": 0}

    metadata = build_metadata(
        schema_version="1.0.0",
        run_info=_build_run_info(config, started_at=started_at, finished_at=_utc_now_iso(), status="success"),
        input_video=video_info,
        roi=clipped_roi_cfg,
        tracking=config.get("tracking", {}),
        event_params=config.get("event", {}),
        summary=summary,
        events=events,
    )
    write_metadata(metadata, metadata_path)
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
