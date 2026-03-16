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


def _longest_true_segment(flags: List[bool]) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    start = -1
    for i, v in enumerate(flags):
        if v and start < 0:
            start = i
        if (not v or i == len(flags) - 1) and start >= 0:
            end = i if v and i == len(flags) - 1 else i - 1
            if end >= start and (best is None or (end - start) > (best[1] - best[0])):
                best = (start, end)
            start = -1
    return best


def _build_track_features(assignments: List[TrackAssignment]) -> Dict[int, Dict[str, float]]:
    feats: Dict[int, Dict[str, float]] = {}
    for a in assignments:
        x1, y1, x2, y2 = a.detection.bbox_xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        cx, cy = _bbox_center(a.detection.bbox_xyxy)
        f = feats.setdefault(a.track_id, {
            "points": [],
            "areas": [],
            "heights": [],
            "center_x": [],
            "center_y": [],
        })
        f["points"].append((cx, cy))
        f["areas"].append(area)
        f["heights"].append(h)
        f["center_x"].append(cx)
        f["center_y"].append(cy)
    return feats


def _auto_detect_active_frame_roi(video_path: Path, width: int, height: int, cfg: Dict[str, Any], template: Optional[Dict[str, Any]], template_match: Dict[str, Any]) -> Dict[str, Any]:
    auto_cfg = cfg.get("calibration", {}).get("active_frame_auto", {})
    sample_frames = int(auto_cfg.get("sample_frames", 40))
    black_th = float(auto_cfg.get("black_col_threshold", 10))
    min_ratio = float(auto_cfg.get("min_content_width_ratio", 0.34))
    max_ratio = float(auto_cfg.get("max_content_width_ratio", 0.94))
    max_jitter = float(auto_cfg.get("max_width_jitter_px", 32))
    min_non_black_ratio = float(auto_cfg.get("min_non_black_ratio", 0.06))
    min_local_variance = float(auto_cfg.get("min_local_variance", 5.0))
    min_edge_density = float(auto_cfg.get("min_edge_density", 0.025))
    min_run_width_ratio = float(auto_cfg.get("min_continuous_width_ratio", 0.34))
    min_acceptable_narrow_width_ratio = float(auto_cfg.get("min_acceptable_narrow_width_ratio", 0.24))
    min_stable_frames = int(auto_cfg.get("min_stable_frames", 8))
    narrow_stable_max_jitter_px = float(auto_cfg.get("narrow_stable_max_jitter_px", 24))
    edge_inset_px = int(auto_cfg.get("edge_inset_px", 8))
    edge_inset_max_px = int(auto_cfg.get("edge_inset_max_px", 24))
    left_q = float(auto_cfg.get("left_quantile_conservative", 0.78))
    right_q = float(auto_cfg.get("right_quantile_conservative", 0.22))

    center_offset_max_ratio = float(auto_cfg.get("center_offset_max_ratio", 0.10))
    symmetry_tolerance_ratio = float(auto_cfg.get("symmetry_tolerance_ratio", 0.28))

    dyn_w_ratio = float(auto_cfg.get("dynamic_default_width_ratio", 0.62))
    dyn_side_ratio = float(auto_cfg.get("dynamic_default_side_margin_ratio", 0.18))
    dyn_min_side_px = int(auto_cfg.get("dynamic_default_min_side_margin_px", 120))
    frame_debug_limit = int(auto_cfg.get("frame_debug_limit", 80))

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    lefts: List[int] = []
    rights: List[int] = []
    widths: List[int] = []
    narrow_lefts: List[int] = []
    narrow_rights: List[int] = []
    narrow_widths: List[int] = []
    sampled = 0
    frame_debug: List[Dict[str, Any]] = []

    if cap.isOpened() and total > 0:
        n = max(1, sample_frames)
        for i in range(n):
            idx = int(i * max(1, total - 1) / max(1, n - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if len(frame_debug) < frame_debug_limit:
                    frame_debug.append({"frame_index": idx, "valid": False, "invalid_reason": "read_failed"})
                continue
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            non_black_ratio = (gray > black_th).mean(axis=0)
            local_variance = gray.var(axis=0)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(gx, gy)
            edge_density = (grad_mag > 22.0).mean(axis=0)

            valid_flags = [
                (non_black_ratio[j] >= min_non_black_ratio)
                and (local_variance[j] >= min_local_variance)
                and (edge_density[j] >= min_edge_density)
                for j in range(gray.shape[1])
            ]
            seg = _longest_true_segment(valid_flags)
            if seg is None:
                if len(frame_debug) < frame_debug_limit:
                    frame_debug.append({"frame_index": idx, "valid": False, "invalid_reason": "no_content_segment", "min_width_check_passed": False})
                continue
            l, r = seg
            seg_w = r - l + 1
            seg_ratio = seg_w / max(1, width)

            inset = min(edge_inset_px, edge_inset_max_px, max(0, seg_w // 8))
            l2, r2 = l + inset, r - inset
            if r2 <= l2:
                if len(frame_debug) < frame_debug_limit:
                    frame_debug.append({"frame_index": idx, "valid": False, "left_candidate": int(l), "right_candidate": int(r), "invalid_reason": "inset_overflow", "min_width_check_passed": False})
                continue

            refined_w = r2 - l2 + 1
            min_width_pass = seg_ratio >= min_run_width_ratio
            narrow_acceptable = (seg_ratio < min_run_width_ratio) and (seg_ratio >= min_acceptable_narrow_width_ratio)

            if min_width_pass:
                lefts.append(l2)
                rights.append(r2)
                widths.append(refined_w)
            elif narrow_acceptable:
                narrow_lefts.append(l2)
                narrow_rights.append(r2)
                narrow_widths.append(refined_w)

            if len(frame_debug) < frame_debug_limit:
                frame_debug.append({
                    "frame_index": idx,
                    "valid": bool(min_width_pass or narrow_acceptable),
                    "left_candidate": int(l2),
                    "right_candidate": int(r2),
                    "segment_width": int(refined_w),
                    "segment_width_ratio": round(seg_ratio, 4),
                    "min_width_check_passed": bool(min_width_pass),
                    "narrow_acceptable": bool(narrow_acceptable),
                    "invalid_reason": None if (min_width_pass or narrow_acceptable) else "min_continuous_width_failed",
                })
    cap.release()

    warnings: List[str] = []
    validity = "failed_no_valid_segments"

    def _evaluate_roi(raw_roi: ROI) -> Tuple[str, Dict[str, float]]:
        ratio = raw_roi[2] / max(1, width)
        content_center = raw_roi[0] + raw_roi[2] / 2.0
        center_offset_ratio = abs(content_center - (width / 2.0)) / max(1.0, width)
        left_bar = raw_roi[0]
        right_bar = max(0, width - (raw_roi[0] + raw_roi[2]))
        symmetry_ratio = abs(left_bar - right_bar) / max(1.0, width)
        stats = {
            "center_offset_ratio": round(center_offset_ratio, 4),
            "side_symmetry_ratio": round(symmetry_ratio, 4),
            "final_width_ratio": round(ratio, 4),
        }
        if ratio < min_acceptable_narrow_width_ratio:
            return "too_narrow", stats
        if ratio > max_ratio:
            return "too_wide", stats
        if center_offset_ratio > center_offset_max_ratio:
            return "off_center", stats
        if symmetry_ratio > symmetry_tolerance_ratio:
            return "asymmetric_side_bars", stats
        return "ok", stats

    detected_candidate_roi = None
    rejected_candidate_info: Dict[str, Any] = {}
    acceptance_mode = "fallback"
    fallback_source: Optional[str] = None

    # Priority 1: normal width accepted candidates
    if widths:
        left = int(_quantile([float(x) for x in lefts], left_q, median(lefts)))
        right = int(_quantile([float(x) for x in rights], right_q, median(rights)))
        raw = (left, 0, max(0, right - left + 1), height)
        detected_candidate_roi = raw
        jitter = float(max(widths) - min(widths)) if widths else 0.0
        pass_ratio = len(widths) / max(1, sampled)
        geo_status, geo_stats = _evaluate_roi(raw)
        if jitter <= max_jitter and pass_ratio >= 0.3 and geo_status == "ok":
            validity = "ok"
            acceptance_mode = "direct"
            roi_clipped, _ = _clip_roi_to_bounds(raw, width, height)
            return {
                "roi_raw": _roi_to_cfg(raw),
                "roi_clipped": _roi_to_cfg(roi_clipped),
                "validity": validity,
                "acceptance_mode": acceptance_mode,
                "fallback_used": False,
                "fallback_source": None,
                "warnings": warnings,
                "detected_candidates": {
                    "primary_candidate_roi": _roi_to_cfg(raw),
                    "candidate_type": "normal_width",
                },
                "rejected_by_threshold": None,
                "final_applied_roi": _roi_to_cfg(roi_clipped),
                "stats": {
                    "sampled_frames": sampled,
                    "valid_frames": len(widths),
                    "valid_ratio": round(pass_ratio, 3),
                    "width_jitter_px": round(jitter, 2),
                    "edge_inset_px_applied": min(edge_inset_px, edge_inset_max_px),
                    **geo_stats,
                },
                "sampling_debug": {
                    "frame_candidates": frame_debug,
                    "invalid_frame_count": len([x for x in frame_debug if not x.get("valid", False)]),
                    "invalid_frame_indices": [int(x.get("frame_index", -1)) for x in frame_debug if not x.get("valid", False) and x.get("frame_index", -1) >= 0],
                },
            }
        rejected_candidate_info = {
            "reason": geo_status if geo_status != "ok" else ("unstable_jitter" if jitter > max_jitter else "low_valid_frame_ratio"),
            "candidate_roi": _roi_to_cfg(raw),
            "candidate_type": "normal_width",
            "jitter_px": round(jitter, 2),
            "valid_ratio": round(pass_ratio, 3),
        }

    # Priority 2: narrow but stable candidates accepted as success
    if narrow_widths:
        nleft = int(_quantile([float(x) for x in narrow_lefts], left_q, median(narrow_lefts)))
        nright = int(_quantile([float(x) for x in narrow_rights], right_q, median(narrow_rights)))
        nraw = (nleft, 0, max(0, nright - nleft + 1), height)
        detected_candidate_roi = nraw
        njitter = float(max(narrow_widths) - min(narrow_widths)) if narrow_widths else 0.0
        nratio = len(narrow_widths) / max(1, sampled)
        geo_status, geo_stats = _evaluate_roi(nraw)
        if len(narrow_widths) >= min_stable_frames and njitter <= narrow_stable_max_jitter_px and nratio >= 0.2 and geo_status == "ok":
            validity = "ok_narrow_stable"
            acceptance_mode = "threshold_relaxed_accept"
            roi_clipped, _ = _clip_roi_to_bounds(nraw, width, height)
            warnings.append("accepted narrow but stable active-frame candidate via relaxed threshold")
            return {
                "roi_raw": _roi_to_cfg(nraw),
                "roi_clipped": _roi_to_cfg(roi_clipped),
                "validity": validity,
                "acceptance_mode": acceptance_mode,
                "fallback_used": False,
                "fallback_source": None,
                "warnings": warnings,
                "detected_candidates": {
                    "primary_candidate_roi": _roi_to_cfg(nraw),
                    "candidate_type": "narrow_stable",
                },
                "rejected_by_threshold": rejected_candidate_info or None,
                "final_applied_roi": _roi_to_cfg(roi_clipped),
                "stats": {
                    "sampled_frames": sampled,
                    "valid_frames": len(narrow_widths),
                    "valid_ratio": round(nratio, 3),
                    "width_jitter_px": round(njitter, 2),
                    "edge_inset_px_applied": min(edge_inset_px, edge_inset_max_px),
                    **geo_stats,
                    "narrow_stable_min_frames_required": min_stable_frames,
                    "narrow_stable_max_jitter_px": narrow_stable_max_jitter_px,
                },
                "sampling_debug": {
                    "frame_candidates": frame_debug,
                    "invalid_frame_count": len([x for x in frame_debug if not x.get("valid", False)]),
                    "invalid_frame_indices": [int(x.get("frame_index", -1)) for x in frame_debug if not x.get("valid", False) and x.get("frame_index", -1) >= 0],
                },
            }
        rejected_candidate_info = {
            "reason": geo_status if geo_status != "ok" else "narrow_candidate_not_stable_enough",
            "candidate_roi": _roi_to_cfg(nraw),
            "candidate_type": "narrow_stable",
            "jitter_px": round(njitter, 2),
            "valid_frames": len(narrow_widths),
        }

    # True fallback path only
    fallback_roi = None
    if template_match.get("matched") and template is not None:
        t_roi = template.get("rois", {}).get("active_frame_roi")
        if t_roi:
            fallback_roi = (int(t_roi["x"]), int(t_roi["y"]), int(t_roi["w"]), int(t_roi["h"]))
            fallback_source = "template"

    if fallback_roi is None:
        side_margin = max(dyn_min_side_px, int(width * dyn_side_ratio))
        target_w = int(width * dyn_w_ratio)
        if target_w <= 0 or target_w > width - 2:
            target_w = int(width * 0.62)
        x = max(side_margin, (width - target_w) // 2)
        right_edge = min(width, width - side_margin)
        w = max(1, right_edge - x)
        if w > target_w:
            x += (w - target_w) // 2
            w = target_w
        fallback_roi = (x, 0, max(1, w), height)
        fallback_source = "dynamic_conservative_default"

    roi_clipped, _ = _clip_roi_to_bounds(fallback_roi, width, height)
    warnings.append(f"active_frame_auto validity={validity}, fallback to {fallback_source}")
    return {
        "roi_raw": _roi_to_cfg(fallback_roi),
        "roi_clipped": _roi_to_cfg(roi_clipped),
        "validity": validity,
        "acceptance_mode": "fallback",
        "fallback_used": True,
        "fallback_source": fallback_source,
        "fallback_reason": validity,
        "warnings": warnings,
        "detected_candidates": {
            "primary_candidate_roi": _roi_to_cfg(detected_candidate_roi) if detected_candidate_roi is not None else None,
            "candidate_type": "normal_or_narrow",
        },
        "rejected_by_threshold": rejected_candidate_info or {
            "reason": "failed_no_valid_segments",
            "candidate_roi": None,
            "candidate_type": "none",
        },
        "final_applied_roi": _roi_to_cfg(roi_clipped),
        "stats": {
            "sampled_frames": sampled,
            "valid_frames": len(widths),
            "narrow_valid_frames": len(narrow_widths),
            "edge_inset_px_applied": min(edge_inset_px, edge_inset_max_px),
            "dynamic_default_policy": {
                "dynamic_default_width_ratio": dyn_w_ratio,
                "dynamic_default_side_margin_ratio": dyn_side_ratio,
                "dynamic_default_min_side_margin_px": dyn_min_side_px,
            },
        },
        "sampling_debug": {
            "frame_candidates": frame_debug,
            "invalid_frame_count": len([x for x in frame_debug if not x.get("valid", False)]),
            "invalid_frame_indices": [int(x.get("frame_index", -1)) for x in frame_debug if not x.get("valid", False) and x.get("frame_index", -1) >= 0],
        },
    }
def _infer_direction_from_tracks(track_features: Dict[int, Dict[str, Any]], active_roi: ROI, core_roi: ROI, cfg: Dict[str, Any], template: Optional[Dict[str, Any]], active_result: Dict[str, Any]) -> Dict[str, Any]:
    mode = cfg.get("template", {}).get("direction_mode", "auto")
    default_dir = cfg.get("event", {}).get("direction", "left_to_right")
    dir_cfg = cfg.get("calibration", {}).get("direction_auto", {})

    if mode == "fixed":
        value = cfg.get("template", {}).get("direction_value") or default_dir
        return {
            "mode": mode,
            "reliable": True,
            "auto_inference": {"status": "ok", "value": value},
            "final": {"value": value, "source": "fixed"},
            "fallback_chain": [{"step": 1, "source": "fixed", "result": "used", "value": value}],
            "warnings": [],
            "explainability": {
                "total_tracks": len(track_features), "filtered_out_tracks": 0, "voting_tracks": 0,
                "left_to_right_score": 0.0, "right_to_left_score": 0.0, "final_confidence": 1.0,
            },
        }

    active_unstable = active_result.get("fallback_used", False) or active_result.get("validity") not in ("ok", "ok_narrow_stable")
    ax, ay, aw, ah = active_roi
    cx, _, cw, _ = core_roi
    total_tracks = len(track_features)

    min_track_points = int(dir_cfg.get("min_track_points", 5))
    min_move_px = float(dir_cfg.get("min_move_px", 60))
    min_area = float(dir_cfg.get("min_bbox_area_px", 2000))
    min_height = float(dir_cfg.get("min_bbox_height_px", 58))
    min_bottom_ratio = float(dir_cfg.get("min_bottom_region_ratio", 0.58))
    max_core_distance_ratio = float(dir_cfg.get("max_core_distance_ratio", 0.28))
    min_consistency = float(dir_cfg.get("min_direction_consistency", 0.70))
    min_voting_tracks = int(dir_cfg.get("min_voting_tracks", 2))
    min_final_confidence = float(dir_cfg.get("min_final_confidence", 0.68))

    candidates: List[Tuple[int, float, float]] = []
    for tid, feat in track_features.items():
        pts = feat.get("points", [])
        if len(pts) < min_track_points:
            continue
        areas = feat.get("areas", [])
        heights = feat.get("heights", [])
        if not areas or not heights:
            continue

        med_area = float(median(areas))
        med_h = float(median(heights))
        med_y = float(median([p[1] for p in pts]))
        delta_x = float(pts[-1][0] - pts[0][0])
        move_abs = abs(delta_x)
        deltas = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
        if not deltas:
            continue
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        consistency = max(pos, neg) / max(1, pos + neg)
        core_dist_ratio = abs(float(median([p[0] for p in pts])) - (cx + cw / 2.0)) / max(1.0, aw)

        if med_area < min_area or med_h < min_height:
            continue
        if med_y < ay + ah * min_bottom_ratio:
            continue
        if core_dist_ratio > max_core_distance_ratio:
            continue
        if move_abs < min_move_px or consistency < min_consistency:
            continue

        weight = (med_area / max(1.0, aw * ah)) * 2.5 + (move_abs / max(1.0, aw)) + (1.0 - core_dist_ratio)
        candidates.append((tid, delta_x, max(0.01, weight)))

    l2r_score = sum(w for _, dx, w in candidates if dx > 0)
    r2l_score = sum(w for _, dx, w in candidates if dx < 0)
    total_score = l2r_score + r2l_score
    confidence = max(l2r_score, r2l_score) / max(1e-6, total_score) if total_score > 0 else 0.0

    reliable = (not active_unstable) and len(candidates) >= min_voting_tracks and total_score > 0 and confidence >= min_final_confidence
    chain = [{"step": 1, "source": "auto_inference", "result": "failed", "value": None}]
    warnings: List[str] = []

    if active_unstable:
        warnings.append("Direction auto downgraded because active_frame_auto is unstable/fallback.")

    if reliable:
        inferred = "left_to_right" if l2r_score >= r2l_score else "right_to_left"
        chain[0] = {"step": 1, "source": "auto_inference", "result": "used", "value": inferred}
        return {
            "mode": "auto",
            "reliable": True,
            "auto_inference": {"status": "ok", "value": inferred},
            "final": {"value": inferred, "source": "auto"},
            "fallback_chain": chain,
            "warnings": warnings,
            "explainability": {
                "total_tracks": total_tracks,
                "filtered_out_tracks": max(0, total_tracks - len(candidates)),
                "voting_tracks": len(candidates),
                "left_to_right_score": round(l2r_score, 3),
                "right_to_left_score": round(r2l_score, 3),
                "final_confidence": round(confidence, 3),
            },
        }

    status = "failed_insufficient_candidates" if len(candidates) < min_voting_tracks else ("unstable" if active_unstable else "failed")
    if status == "failed_insufficient_candidates":
        warnings.append("Direction auto failed: insufficient candidate tracks for voting.")

    t_dir = template.get("direction", {}).get("direction_value") if template is not None else None
    if t_dir in ("left_to_right", "right_to_left"):
        chain.append({"step": 2, "source": "template_direction", "result": "used", "value": t_dir})
        warnings.append("Direction auto unreliable; fallback to template direction.")
        final_value, final_source = t_dir, "template_direction"
    else:
        chain.append({"step": 2, "source": "template_direction", "result": "unavailable", "value": None})
        chain.append({"step": 3, "source": "default_direction", "result": "used", "value": default_dir})
        warnings.append("Direction auto unreliable and template direction unavailable; fallback to default direction.")
        final_value, final_source = default_dir, "default_direction"

    return {
        "mode": "auto",
        "reliable": False,
        "auto_inference": {"status": status, "value": None},
        "final": {"value": final_value, "source": final_source},
        "fallback_chain": chain,
        "warnings": warnings,
        "explainability": {
            "total_tracks": total_tracks,
            "filtered_out_tracks": max(0, total_tracks - len(candidates)),
            "voting_tracks": len(candidates),
            "left_to_right_score": round(l2r_score, 3),
            "right_to_left_score": round(r2l_score, 3),
            "final_confidence": round(min(confidence, 0.55) if active_unstable else confidence, 3),
        },
    }

def _suggest_rois_from_tracks(track_features: Dict[int, Dict[str, Any]], active_roi: ROI, direction: str, direction_reliable: bool, cfg: Dict[str, Any], template: Optional[Dict[str, Any]], template_match: Dict[str, Any]) -> Dict[str, Any]:
    rs_cfg = cfg.get("calibration", {}).get("roi_suggestion", {})
    min_valid_tracks = int(rs_cfg.get("min_valid_tracks", 3))

    template_geo = rs_cfg.get("template_geometry", {})
    entry_ratio = template_geo.get("entry_ratio", [0.06, 0.20])
    core_ratio = template_geo.get("core_ratio", [0.34, 0.62])
    exit_ratio = template_geo.get("exit_ratio", [0.78, 0.94])
    max_core_width_ratio = float(template_geo.get("max_core_width_ratio", 0.45))
    safe_border_margin_px = int(template_geo.get("safe_border_margin_px", 24))

    adjust_cfg = rs_cfg.get("micro_adjust", {})
    max_shift_px = int(adjust_cfg.get("max_shift_px", 36))
    max_expand_ratio = float(adjust_cfg.get("max_expand_ratio", 0.12))

    guard = rs_cfg.get("min_size_guard", {})
    min_w = int(guard.get("min_roi_width_px", 80))
    min_h = int(guard.get("min_roi_height_px", 120))

    vertical_cfg = rs_cfg.get("vertical_padding", {})
    headroom_px = int(vertical_cfg.get("headroom_px", 50))
    footroom_px = int(vertical_cfg.get("footroom_px", 35))
    min_height_ratio = float(vertical_cfg.get("min_height_ratio", 0.38))

    ax, ay, aw, ah = active_roi

    # Existing template priority when direction is unreliable.
    if not direction_reliable and template_match.get("matched") and template is not None:
        troi = template.get("rois", {})
        if all(k in troi for k in ("entry_roi", "core_roi", "exit_roi")):
            e = (int(troi["entry_roi"]["x"]), int(troi["entry_roi"]["y"]), int(troi["entry_roi"]["w"]), int(troi["entry_roi"]["h"]))
            c = (int(troi["core_roi"]["x"]), int(troi["core_roi"]["y"]), int(troi["core_roi"]["w"]), int(troi["core_roi"]["h"]))
            x = (int(troi["exit_roi"]["x"]), int(troi["exit_roi"]["y"]), int(troi["exit_roi"]["w"]), int(troi["exit_roi"]["h"]))
            e = _clip_roi_to_parent(e, active_roi)[0]
            c = _clip_roi_to_parent(c, active_roi)[0]
            x = _clip_roi_to_parent(x, active_roi)[0]
            t = _build_tracking_roi([e, c, x], int(cfg.get("tracking", {}).get("tracking_roi_margin_px", 50)), active_roi)
            return {
                "mode": "template_from_existing_camera",
                "result": {"entry_roi": _roi_to_cfg(e), "core_roi": _roi_to_cfg(c), "exit_roi": _roi_to_cfg(x), "tracking_roi": _roi_to_cfg(t)},
                "scores": {"overall_confidence_score": 0.62, "track_count_score": 0.5, "direction_consistency_score": 0.5, "path_compactness_score": 0.5, "geometry_penalty": 0.0},
                "warnings": ["direction unreliable; using template_from_existing_camera."],
                "min_size_guard": {"min_roi_width_px": min_w, "min_roi_height_px": min_h},
                "vertical_padding": {"headroom_px": headroom_px, "footroom_px": footroom_px},
                "geometry_flags": [],
            }

    # If direction unreliable, force neutral geometry and avoid directional swapping.
    mode = "directional_template"
    if not direction_reliable:
        mode = "neutral_template_due_to_unreliable_direction"

    local_entry_ratio = list(entry_ratio)
    local_exit_ratio = list(exit_ratio)
    if direction_reliable and direction == "right_to_left":
        local_entry_ratio, local_exit_ratio = local_exit_ratio, local_entry_ratio

    y_base = ay + int(ah * 0.38)
    h_base = max(min_h, int(ah * min_height_ratio))

    def mk(seg: List[float], max_w_ratio: Optional[float] = None) -> ROI:
        sx = int(ax + aw * seg[0])
        ex = int(ax + aw * seg[1])
        w = max(min_w, ex - sx)
        if max_w_ratio is not None:
            w = min(w, int(aw * max_w_ratio))
        return _clip_roi_to_parent((sx, y_base, w, h_base), active_roi)[0]

    entry = mk(local_entry_ratio)
    core = mk(core_ratio, max_core_width_ratio)
    exit_roi = mk(local_exit_ratio)

    points: List[Tuple[float, float]] = []
    for feat in track_features.values():
        points.extend(feat.get("points", []))

    warnings: List[str] = []
    if not direction_reliable:
        warnings.append("direction unreliable; switched to neutral template geometry.")

    used_track_count = len([1 for feat in track_features.values() if len(feat.get("points", [])) >= 2])
    if direction_reliable and used_track_count >= min_valid_tracks and len(points) >= 20:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_center = int(_quantile(xs, 0.50, ax + aw / 2.0))
        shift = max(-max_shift_px, min(max_shift_px, x_center - (ax + aw // 2)))

        def shift_and_expand(roi: ROI, allow_expand: bool = True) -> ROI:
            x, y, w, h = roi
            nw = int(w * (1.0 + max_expand_ratio * 0.5)) if allow_expand else w
            return _clip_roi_to_parent((x + shift, y, nw, h), active_roi)[0]

        entry = shift_and_expand(entry)
        core = shift_and_expand(core, allow_expand=False)
        if core[2] > int(aw * max_core_width_ratio):
            core = _clip_roi_to_parent((core[0], core[1], int(aw * max_core_width_ratio), core[3]), active_roi)[0]
        exit_roi = shift_and_expand(exit_roi)

        y_low = int(_quantile(ys, 0.20, ay + int(ah * 0.40))) - headroom_px
        y_high = int(_quantile(ys, 0.88, ay + int(ah * 0.85))) + footroom_px
        y_low = max(ay, y_low)
        y_high = min(ay + ah, y_high)
        height_new = max(min_h, int(ah * min_height_ratio), y_high - y_low)

        entry = _clip_roi_to_parent((entry[0], y_low, entry[2], height_new), active_roi)[0]
        core = _clip_roi_to_parent((core[0], y_low, core[2], height_new), active_roi)[0]
        exit_roi = _clip_roi_to_parent((exit_roi[0], y_low, exit_roi[2], height_new), active_roi)[0]
    elif direction_reliable:
        warnings.append("Insufficient reliable tracks; used conservative directional template with no micro-adjust.")

    geometry_flags: List[str] = []
    if entry[0] < ax + safe_border_margin_px:
        geometry_flags.append("entry_too_close_left_border")
    if exit_roi[0] + exit_roi[2] > ax + aw - safe_border_margin_px:
        geometry_flags.append("exit_too_close_right_border")
    if core[2] > int(aw * max_core_width_ratio):
        geometry_flags.append("core_too_wide")
    if core[3] < int(ah * min_height_ratio):
        geometry_flags.append("core_too_short")
    if geometry_flags:
        warnings.append("ROI geometry sanity warnings: " + ", ".join(geometry_flags))

    tracking_roi = _build_tracking_roi([entry, core, exit_roi], int(cfg.get("tracking", {}).get("tracking_roi_margin_px", 50)), active_roi)

    track_count_score = min(1.0, used_track_count / max(1, min_valid_tracks))
    direction_consistency_score = 0.75 if direction_reliable else 0.45
    path_compactness_score = 0.55 if points else 0.4
    geometry_penalty = 0.12 * len(geometry_flags)
    if not direction_reliable:
        geometry_penalty += 0.08
    overall = max(0.0, (track_count_score + direction_consistency_score + path_compactness_score) / 3.0 - geometry_penalty)

    return {
        "mode": mode,
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
            "geometry_penalty": round(geometry_penalty, 3),
        },
        "warnings": warnings,
        "min_size_guard": {"min_roi_width_px": min_w, "min_roi_height_px": min_h},
        "vertical_padding": {"headroom_px": headroom_px, "footroom_px": footroom_px},
        "geometry_flags": geometry_flags,
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

    track_features: Dict[int, Dict[str, Any]] = {}

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
        frame_feats = _build_track_features(assigns)
        for tid, feat in frame_feats.items():
            acc = track_features.setdefault(tid, {"points": [], "areas": [], "heights": [], "center_x": [], "center_y": []})
            for k in ["points", "areas", "heights", "center_x", "center_y"]:
                acc[k].extend(feat[k])
        fidx += 1

    # direction with fallback chain
    direction_info = _infer_direction_from_tracks(track_features, active_frame_roi, _parse_roi(clipped_roi_cfg, "core_roi"), config, template, active_result)

    # roi suggestion
    roi_suggestion = _suggest_rois_from_tracks(track_features, active_frame_roi, direction_info["final"]["value"], bool(direction_info.get("reliable", False)), config, template, template_match)

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

    manual_reasons: List[Dict[str, str]] = []
    risk_flags: List[str] = []
    if active_result["fallback_used"]:
        risk_flags.append("active_fallback")
        manual_reasons.append({
            "code": "active_frame_fallback",
            "message": f"active_frame_auto fallback used: source={active_result['fallback_source']}, validity={active_result['validity']}"
        })
    if not bool(direction_info.get("reliable", False)):
        risk_flags.append("direction_unreliable")
        ex = direction_info.get("explainability", {})
        manual_reasons.append({
            "code": "direction_unreliable_or_fallback",
            "message": (
                "direction not reliable; "
                f"voting_tracks={ex.get('voting_tracks', 0)}, "
                f"l2r={ex.get('left_to_right_score', 0)}, r2l={ex.get('right_to_left_score', 0)}, "
                f"confidence={ex.get('final_confidence', 0)}, final_source={direction_info.get('final', {}).get('source', 'unknown')}"
            )
        })
    if roi_suggestion.get("mode") != "directional_template":
        risk_flags.append("roi_low_confidence_geometry")
        manual_reasons.append({
            "code": "roi_generated_under_low_confidence_geometry",
            "message": f"roi_suggestion mode={roi_suggestion.get('mode')} due to unreliable direction/context"
        })
    if roi_suggestion["geometry_flags"]:
        risk_flags.append("roi_geometry_anomaly")
        manual_reasons.append({
            "code": "roi_geometry_anomaly",
            "message": "roi_geometry anomalies: " + ", ".join(roi_suggestion["geometry_flags"])
        })
    if roi_suggestion["scores"]["overall_confidence_score"] < 0.65:
        risk_flags.append("roi_low_score")
        manual_reasons.append({
            "code": "roi_low_confidence_score",
            "message": f"roi_suggestion overall_confidence_score low: {roi_suggestion['scores']['overall_confidence_score']}"
        })

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
            "final_applied_roi_is_authoritative": True,
            "result": active_result,
        },
        "direction": direction_info,
        "roi_suggestion": {
            "enabled": bool(config.get("calibration", {}).get("roi_suggestion", {}).get("enabled", True)),
            "mode": roi_suggestion["mode"],
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
            "priority": "high" if ("active_fallback" in risk_flags or "direction_unreliable" in risk_flags or "roi_low_confidence_geometry" in risk_flags or "roi_geometry_anomaly" in risk_flags or len(risk_flags) >= 2) else ("medium" if len(risk_flags) == 1 else "low"),
            "reasons": manual_reasons,
            "suggested_actions": [
                "Confirm active_frame_roi left/right boundaries and black-border exclusion parameters.",
                "Confirm direction and adjust calibration.direction_auto thresholds if needed.",
                "Confirm ROI geometry (entry/core/exit) and vertical padding headroom/footroom.",
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
