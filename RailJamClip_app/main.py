"""Main entry for RailJamClip_app MVP.

本版重点：最小可运行自动标定
- 自动检测 active_frame_roi（失败回退）
- 短时检测+tracking 生成 entry/core/exit 初始建议
- 生成 calibration_preview.jpg
- 生成 calibration_report.json（required 结构）
- 保持 metadata.json 顶层结构不变
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

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


def _warn_if_small_roi(name: str, roi: ROI, frame_w: int, frame_h: int, min_w: int, min_h: int, min_area_ratio: float) -> Optional[str]:
    _, _, rw, rh = roi
    area = rw * rh
    area_threshold = frame_w * frame_h * min_area_ratio
    if rw < min_w or rh < min_h or area < area_threshold:
        msg = f"ROI '{name}' may be too small after clipping: w={rw}, h={rh}, area={area}, area_threshold={int(area_threshold)}"
        print(f"[WARN] {msg}")
        return msg
    return None


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

    ax, ay, aw, ah = active
    cv2.putText(frame, f"active_frame_roi: x={ax}, y={ay}, w={aw}, h={ah}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"[INFO] Calibration preview saved: {out_path}")


def _quantile(values: List[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _detect_aspect_hint(width: int, height: int) -> str:
    ratio = width / max(1, height)
    if ratio < 0.8:
        return "portrait"
    if ratio > 1.2:
        return "landscape"
    return "near_square"


def _load_template(template_path: Path) -> Optional[Dict[str, Any]]:
    if not template_path.exists():
        return None
    try:
        with template_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _template_match_check(template: Optional[Dict[str, Any]], width: int, height: int, aspect_hint: str) -> Dict[str, Any]:
    if template is None:
        return {
            "matched": False,
            "reason": "template_not_found",
            "criteria": {
                "require_exact_resolution": True,
                "require_aspect_hint_match": True,
                "template_resolution": None,
                "current_resolution": [width, height],
                "template_aspect_hint": None,
                "current_aspect_hint": aspect_hint,
            },
        }

    t_w = int(template.get("video_profile", {}).get("width", -1))
    t_h = int(template.get("video_profile", {}).get("height", -1))
    t_aspect = template.get("video_profile", {}).get("aspect_hint")

    matched = (t_w == width and t_h == height and t_aspect == aspect_hint)
    reason = "ok" if matched else "resolution_or_aspect_mismatch"
    return {
        "matched": matched,
        "reason": reason,
        "criteria": {
            "require_exact_resolution": True,
            "require_aspect_hint_match": True,
            "template_resolution": [t_w, t_h],
            "current_resolution": [width, height],
            "template_aspect_hint": t_aspect,
            "current_aspect_hint": aspect_hint,
        },
    }


def _auto_detect_active_frame_roi(video_path: Path, width: int, height: int, cfg: Dict[str, Any], template: Optional[Dict[str, Any]], template_match: Dict[str, Any]) -> Dict[str, Any]:
    auto_cfg = cfg.get("calibration", {}).get("active_frame_auto", {})
    sample_frames = int(auto_cfg.get("sample_frames", 40))
    black_th = float(auto_cfg.get("black_col_threshold", 10))
    min_ratio = float(auto_cfg.get("min_content_width_ratio", 0.45))
    max_ratio = float(auto_cfg.get("max_content_width_ratio", 0.98))
    max_jitter = float(auto_cfg.get("max_width_jitter_px", 40))

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    lefts: List[int] = []
    rights: List[int] = []
    widths: List[int] = []

    if cap.isOpened() and total > 0:
        n = max(1, sample_frames)
        for i in range(n):
            idx = int(i * max(1, total - 1) / max(1, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            col_mean = gray.mean(axis=0)
            valid_cols = [j for j, v in enumerate(col_mean) if v > black_th]
            if not valid_cols:
                continue
            l, r = min(valid_cols), max(valid_cols)
            lefts.append(l)
            rights.append(r)
            widths.append(r - l + 1)
    cap.release()

    warnings: List[str] = []
    validity = "ok"
    fallback_used = False
    fallback_source = "none"

    if not widths:
        validity = "failed"
    else:
        left = int(median(lefts))
        right = int(median(rights))
        raw = (left, 0, max(0, right - left + 1), height)
        ratio = raw[2] / max(1, width)
        jitter = float(max(widths) - min(widths)) if widths else 0.0
        if ratio < min_ratio:
            validity = "too_narrow"
        elif ratio > max_ratio:
            validity = "too_wide"
        elif jitter > max_jitter:
            validity = "unstable"
        roi_raw = raw
        roi_clipped, _ = _clip_roi_to_bounds(roi_raw, width, height)

        if validity == "ok":
            return {
                "roi_raw": _roi_to_cfg(roi_raw),
                "roi_clipped": _roi_to_cfg(roi_clipped),
                "validity": validity,
                "fallback_used": False,
                "fallback_source": "none",
                "warnings": warnings,
            }

    # fallback
    fallback_used = True
    fallback_roi = None
    if template_match.get("matched") and template is not None:
        t_roi = template.get("rois", {}).get("active_frame_roi")
        if t_roi:
            fallback_roi = (int(t_roi["x"]), int(t_roi["y"]), int(t_roi["w"]), int(t_roi["h"]))
            fallback_source = "template"

    if fallback_roi is None:
        d = auto_cfg.get("default_active_frame_roi", {"x": 120, "y": 0, "w": max(1, int(width * 0.8)), "h": height})
        fallback_roi = (int(d.get("x", 0)), int(d.get("y", 0)), int(d.get("w", width)), int(d.get("h", height)))
        fallback_source = "default"

    fallback_roi_clipped, _ = _clip_roi_to_bounds(fallback_roi, width, height)
    warnings.append(f"active_frame_auto validity={validity}, fallback to {fallback_source}")
    return {
        "roi_raw": _roi_to_cfg(fallback_roi),
        "roi_clipped": _roi_to_cfg(fallback_roi_clipped),
        "validity": validity,
        "fallback_used": fallback_used,
        "fallback_source": fallback_source,
        "warnings": warnings,
    }


def _infer_direction_from_tracks(track_samples: Dict[int, List[Tuple[float, float]]], cfg: Dict[str, Any], template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    mode = cfg.get("template", {}).get("direction_mode", "auto")
    default_dir = cfg.get("event", {}).get("direction", "left_to_right")

    if mode == "fixed":
        value = cfg.get("template", {}).get("direction_value") or default_dir
        return {
            "mode": mode,
            "auto_inference": {"status": "ok", "value": value},
            "final": {"value": value, "source": "fixed"},
            "fallback_chain": [{"step": 1, "source": "fixed", "result": "used", "value": value}],
            "warnings": [],
        }

    # auto mode
    deltas: List[float] = []
    for pts in track_samples.values():
        if len(pts) < 2:
            continue
        deltas.append(pts[-1][0] - pts[0][0])

    chain = [{"step": 1, "source": "auto_inference", "result": "failed", "value": None}]
    warnings: List[str] = []

    if len(deltas) >= 2:
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        total = max(1, pos + neg)
        consistency = max(pos, neg) / total
        if consistency >= 0.6:
            inferred = "left_to_right" if pos >= neg else "right_to_left"
            chain[0] = {"step": 1, "source": "auto_inference", "result": "used", "value": inferred}
            return {
                "mode": "auto",
                "auto_inference": {"status": "ok", "value": inferred},
                "final": {"value": inferred, "source": "auto"},
                "fallback_chain": chain,
                "warnings": warnings,
            }

    # fallback: template direction -> default
    t_dir = None
    if template is not None:
        t_dir = template.get("direction", {}).get("direction_value")
    if t_dir in ("left_to_right", "right_to_left"):
        chain.append({"step": 2, "source": "template_direction", "result": "used", "value": t_dir})
        warnings.append("Direction auto inference failed/unstable; fallback to template direction.")
        return {
            "mode": "auto",
            "auto_inference": {"status": "failed", "value": None},
            "final": {"value": t_dir, "source": "template_direction"},
            "fallback_chain": chain,
            "warnings": warnings,
        }

    chain.append({"step": 2, "source": "template_direction", "result": "unavailable", "value": None})
    chain.append({"step": 3, "source": "default_direction", "result": "used", "value": default_dir})
    warnings.append("Direction auto inference failed/unstable; template direction unavailable; fallback to default direction.")
    return {
        "mode": "auto",
        "auto_inference": {"status": "failed", "value": None},
        "final": {"value": default_dir, "source": "default_direction"},
        "fallback_chain": chain,
        "warnings": warnings,
    }


def _suggest_rois_from_tracks(track_samples: Dict[int, List[Tuple[float, float]]], active_roi: ROI, direction: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    rs_cfg = cfg.get("calibration", {}).get("roi_suggestion", {})
    min_valid_tracks = int(rs_cfg.get("min_valid_tracks", 3))
    entry_ratio = rs_cfg.get("entry_ratio", [0.05, 0.25])
    core_ratio = rs_cfg.get("core_ratio", [0.30, 0.70])
    exit_ratio = rs_cfg.get("exit_ratio", [0.75, 0.95])

    guard = rs_cfg.get("min_size_guard", {})
    min_w = int(guard.get("min_roi_width_px", 80))
    min_h = int(guard.get("min_roi_height_px", 100))

    points: List[Tuple[float, float]] = []
    for pts in track_samples.values():
        points.extend(pts)

    ax, ay, aw, ah = active_roi

    # fallback defaults from config if poor data
    raw_roi_cfg = cfg.get("roi", {})
    fallback_entry = _parse_roi(raw_roi_cfg, "entry_roi")
    fallback_core = _parse_roi(raw_roi_cfg, "core_roi")
    fallback_exit = _parse_roi(raw_roi_cfg, "exit_roi")

    warnings: List[str] = []
    if len(track_samples) < min_valid_tracks or len(points) < 20:
        warnings.append("Insufficient track samples for robust ROI suggestion; fallback to configured stage ROIs clipped to active_frame_roi.")
        e, _ = _clip_roi_to_parent(fallback_entry, active_roi)
        c, _ = _clip_roi_to_parent(fallback_core, active_roi)
        x, _ = _clip_roi_to_parent(fallback_exit, active_roi)
        t = _build_tracking_roi([e, c, x], int(cfg.get("tracking", {}).get("tracking_roi_margin_px", 50)), active_roi)
        return {
            "result": {"entry_roi": _roi_to_cfg(e), "core_roi": _roi_to_cfg(c), "exit_roi": _roi_to_cfg(x), "tracking_roi": _roi_to_cfg(t)},
            "scores": {
                "overall_confidence_score": 0.4,
                "track_count_score": min(1.0, len(track_samples) / max(1, min_valid_tracks)),
                "direction_consistency_score": 0.4,
                "path_compactness_score": 0.4,
            },
            "warnings": warnings,
            "min_size_guard": {"min_roi_width_px": min_w, "min_roi_height_px": min_h},
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    y1 = int(_quantile(ys, 0.20, ay))
    y2 = int(_quantile(ys, 0.80, ay + ah))
    roi_h = max(min_h, y2 - y1)

    if direction == "right_to_left":
        entry_ratio, exit_ratio = exit_ratio, entry_ratio

    def mk(seg: List[float]) -> ROI:
        sx = int(ax + aw * seg[0])
        ex = int(ax + aw * seg[1])
        w = max(min_w, ex - sx)
        roi = (sx, y1, w, roi_h)
        clipped, _ = _clip_roi_to_parent(roi, active_roi)
        return clipped

    entry = mk(entry_ratio)
    core = mk(core_ratio)
    exit_roi = mk(exit_ratio)
    tracking_roi = _build_tracking_roi([entry, core, exit_roi], int(cfg.get("tracking", {}).get("tracking_roi_margin_px", 50)), active_roi)

    # scores
    track_count_score = min(1.0, len(track_samples) / max(1, min_valid_tracks))

    deltas = []
    for pts in track_samples.values():
        if len(pts) >= 2:
            deltas.append(pts[-1][0] - pts[0][0])
    if deltas:
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        direction_consistency_score = max(pos, neg) / max(1, pos + neg)
    else:
        direction_consistency_score = 0.3

    std_y = (max(ys) - min(ys)) / max(1.0, ah)
    path_compactness_score = max(0.0, 1.0 - min(1.0, std_y))
    overall = (track_count_score + direction_consistency_score + path_compactness_score) / 3.0

    return {
        "result": {
            "entry_roi": _roi_to_cfg(entry),
            "core_roi": _roi_to_cfg(core),
            "exit_roi": _roi_to_cfg(exit_roi),
            "tracking_roi": _roi_to_cfg(tracking_roi),
        },
        "scores": {
            "overall_confidence_score": round(overall, 3),
            "track_count_score": round(track_count_score, 3),
            "direction_consistency_score": round(direction_consistency_score, 3),
            "path_compactness_score": round(path_compactness_score, 3),
        },
        "warnings": warnings,
        "min_size_guard": {"min_roi_width_px": min_w, "min_roi_height_px": min_h},
    }


def _filter_detections_for_tracking(detections: List[Detection], tracking_cfg: Dict[str, Any], active_frame_roi: ROI, tracking_roi: ROI) -> List[Detection]:
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
    aspect_hint = _detect_aspect_hint(frame_w, frame_h)

    print(f"[INFO] 已打开视频: {input_video_path}")
    print(
        "[INFO] 视频基础信息: "
        f"fps={video_info['fps']}, total_frames={video_info['total_frames']}, width={frame_w}, height={frame_h}, "
        f"duration_seconds={video_info['duration_seconds']:.3f}"
    )

    output_cfg = config.get("output", {})
    clips_dir = Path(output_cfg["clips_dir"])
    debug_log_path = Path(output_cfg["debug_log_path"])
    metadata_path = Path(output_cfg["metadata_path"])

    debug_cfg = config.get("debug", {})
    debug_json_enabled = bool(debug_cfg.get("export_detection_json", True))
    debug_json_path = Path(debug_cfg.get("detection_json_path", "output/detections_debug.json"))
    preview_enabled = bool(debug_cfg.get("export_preview_video", True))
    preview_path = Path(debug_cfg.get("preview_video_path", "output/preview_tracking.mp4"))

    calibration_cfg = config.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("export_calibration_preview", True))
    calibration_preview_path = Path(calibration_cfg.get("preview_image_path", "output/calibration_preview.jpg"))
    calibration_report_path = Path(calibration_cfg.get("calibration_report_path", "output/calibration_report.json"))

    ensure_dir(clips_dir)
    ensure_dir(debug_log_path.parent)

    # template loading + matching
    template_cfg = config.get("template", {})
    template_path = Path(template_cfg.get("save_path", "templates/camera_001.json"))
    template = _load_template(template_path)
    template_match = _template_match_check(template, frame_w, frame_h, aspect_hint)

    template_loading = {
        "enabled": bool(template_cfg.get("enabled", True)),
        "template_id_requested": template_cfg.get("template_id", "camera_001"),
        "template_found": template is not None,
        "match_check": template_match,
    }

    # active_frame auto
    active_result = _auto_detect_active_frame_roi(input_video_path, frame_w, frame_h, config, template, template_match)
    active_frame_roi = _parse_roi({"active_frame_roi": active_result["roi_clipped"]}, "active_frame_roi")

    # stage ROI clip from full-frame coords -> active_frame
    roi_cfg = config.get("roi", {})
    tracking_cfg = config.get("tracking", {})

    roi_min_w = int(tracking_cfg.get("min_roi_width_px", 24))
    roi_min_h = int(tracking_cfg.get("min_roi_height_px", 24))
    roi_min_area_ratio = float(tracking_cfg.get("min_roi_area_ratio", 0.001))

    clipped_roi_cfg: Dict[str, Any] = {"active_frame_roi": _roi_to_cfg(active_frame_roi)}
    roi_warnings: List[str] = []
    stage_rois: List[ROI] = []

    for key in ["entry_roi", "core_roi", "exit_roi"]:
        raw = _parse_roi(roi_cfg, key)
        clipped, changed = _clip_roi_to_parent(raw, active_frame_roi)
        if changed:
            msg = f"ROI '{key}' clipped to active_frame_roi from {raw} to {clipped}"
            print(f"[WARN] {msg}")
            roi_warnings.append(msg)
        s = _warn_if_small_roi(key, clipped, frame_w, frame_h, roi_min_w, roi_min_h, roi_min_area_ratio)
        if s:
            roi_warnings.append(s)
        clipped_roi_cfg[key] = _roi_to_cfg(clipped)
        stage_rois.append(clipped)

    # init tracking_roi from stage + margin
    tracking_roi = _build_tracking_roi(stage_rois, int(tracking_cfg.get("tracking_roi_margin_px", 50)), active_frame_roi)
    clipped_roi_cfg["tracking_roi"] = _roi_to_cfg(tracking_roi)

    # open video for calibration frame and short suggestion run
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for detection: {input_video_path}")
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Failed to read first frame for calibration preview")

    print("[INFO] 正在加载 YOLO 模型...")
    detector_cfg = config.get("detector", {})
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

    # short calibration suggestion pass
    frame_step = int(detector_cfg.get("frame_step", 2))
    sample_seconds = int(config.get("calibration", {}).get("roi_suggestion", {}).get("sample_seconds", 20))
    calib_frames_limit = int(max(1, video_info.get("fps", 30.0) * sample_seconds))

    direction_seed = str(config.get("event", {}).get("direction", "left_to_right"))
    tracker_calib = MinimalTracker(
        iou_threshold=float(tracking_cfg.get("iou_threshold", 0.25)),
        max_center_distance_px=float(tracking_cfg.get("max_center_distance_px", 80)),
        max_lost_frames=int(tracking_cfg.get("max_lost_frames", 8)),
        min_track_frames=int(tracking_cfg.get("min_track_frames", 6)),
        motion_min_frames=int(tracking_cfg.get("min_motion_frames", 3)),
        motion_min_distance_px=float(tracking_cfg.get("min_motion_distance_px", 10)),
        direction=direction_seed,
        direction_min_progress_px=float(tracking_cfg.get("direction_min_progress_px", 5)),
        core_roi=_parse_roi(clipped_roi_cfg, "core_roi"),
        core_reacquire_max_frames=int(tracking_cfg.get("core_reacquire_max_frames", 10)),
        core_reacquire_max_dist_px=float(tracking_cfg.get("core_reacquire_max_dist_px", 120)),
    )

    track_samples: Dict[int, List[Tuple[float, float]]] = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fidx = 0
    while fidx < calib_frames_limit:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % frame_step != 0:
            fidx += 1
            continue
        dets = detector.predict_frame(frame)
        filtered = _filter_detections_for_tracking(dets, tracking_cfg, active_frame_roi, tracking_roi)
        assigns = tracker_calib.update(filtered, fidx)
        for a in assigns:
            cx, cy = _bbox_center(a.detection.bbox_xyxy)
            track_samples.setdefault(a.track_id, []).append((cx, cy))
        fidx += 1

    # direction with fallback chain
    direction_info = _infer_direction_from_tracks(track_samples, config, template)

    # roi suggestion
    roi_suggestion = _suggest_rois_from_tracks(track_samples, active_frame_roi, direction_info["final"]["value"], config)

    # apply suggested rois as runtime rois
    clipped_roi_cfg.update(roi_suggestion["result"])
    tracking_roi = _parse_roi(clipped_roi_cfg, "tracking_roi")

    if calibration_enabled:
        _render_calibration_preview(first_frame, clipped_roi_cfg, calibration_preview_path)

    # calibration report (required subset)
    fallback_items: List[Dict[str, Any]] = []
    if active_result["fallback_used"]:
        fallback_items.append(
            {
                "module": "active_frame_auto",
                "trigger": active_result["validity"],
                "fallback_source": active_result["fallback_source"],
                "applied": True,
            }
        )
    if direction_info["final"]["source"] != "auto" and direction_info["mode"] == "auto":
        fallback_items.append(
            {
                "module": "direction",
                "trigger": direction_info["auto_inference"]["status"],
                "fallback_source": direction_info["final"]["source"],
                "applied": True,
            }
        )
    if not template_match["matched"]:
        fallback_items.append(
            {
                "module": "template_loading",
                "trigger": template_match["reason"],
                "fallback_source": "auto_calibration",
                "applied": True,
            }
        )

    manual_reasons: List[str] = []
    if active_result["fallback_used"]:
        manual_reasons.append("active_frame_auto used fallback")
    if direction_info["final"]["source"] != "auto" and direction_info["mode"] == "auto":
        manual_reasons.append("direction fallback path used")
    if roi_suggestion["scores"]["overall_confidence_score"] < 0.7:
        manual_reasons.append("roi_suggestion overall_confidence_score is low")

    calibration_report = {
        "schema_version": "1.0.0",
        "run_id": f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": _utc_now_iso(),
        "input_video": {
            "video_path": str(input_video_path),
            "width": frame_w,
            "height": frame_h,
            "fps": float(video_info["fps"]),
            "total_frames": int(video_info["total_frames"]),
            "duration_seconds": float(video_info["duration_seconds"]),
        },
        "template_loading": template_loading,
        "active_frame_auto": {
            "enabled": bool(config.get("calibration", {}).get("active_frame_auto", {}).get("enabled", True)),
            "result": active_result,
        },
        "direction": direction_info,
        "roi_suggestion": {
            "enabled": bool(config.get("calibration", {}).get("roi_suggestion", {}).get("enabled", True)),
            "min_size_guard": roi_suggestion["min_size_guard"],
            "result": roi_suggestion["result"],
            "scores": roi_suggestion["scores"],
            "warnings": roi_suggestion["warnings"] + roi_warnings,
        },
        "fallback_summary": {
            "used": bool(fallback_items),
            "items": fallback_items,
        },
        "manual_review": {
            "recommended": bool(manual_reasons),
            "priority": "high" if len(manual_reasons) >= 2 else ("medium" if len(manual_reasons) == 1 else "low"),
            "reasons": manual_reasons,
            "suggested_actions": [
                "Check active_frame_roi and side black bars alignment.",
                "Verify entry/core/exit placement on calibration preview.",
            ] if manual_reasons else [],
        },
    }

    write_json(calibration_report, calibration_report_path)
    print(f"[INFO] Calibration report saved: {calibration_report_path}")

    # reset tracker for main run
    tracker = MinimalTracker(
        iou_threshold=float(tracking_cfg.get("iou_threshold", 0.25)),
        max_center_distance_px=float(tracking_cfg.get("max_center_distance_px", 80)),
        max_lost_frames=int(tracking_cfg.get("max_lost_frames", 8)),
        min_track_frames=int(tracking_cfg.get("min_track_frames", 6)),
        motion_min_frames=int(tracking_cfg.get("min_motion_frames", 3)),
        motion_min_distance_px=float(tracking_cfg.get("min_motion_distance_px", 10)),
        direction=direction_info["final"]["value"],
        direction_min_progress_px=float(tracking_cfg.get("direction_min_progress_px", 5)),
        core_roi=_parse_roi(clipped_roi_cfg, "core_roi"),
        core_reacquire_max_frames=int(tracking_cfg.get("core_reacquire_max_frames", 10)),
        core_reacquire_max_dist_px=float(tracking_cfg.get("core_reacquire_max_dist_px", 120)),
    )

    # run main detection/tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
            writer.write(_draw_tracking_overlay(frame.copy(), frame_idx, assignments, clipped_roi_cfg))

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
