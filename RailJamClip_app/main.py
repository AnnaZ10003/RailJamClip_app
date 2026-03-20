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


def _safe_write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _resolve_export_direction_explainability(
    calibration_report: Dict[str, Any],
    direction_info: Dict[str, Any],
) -> Dict[str, Any]:
    report_direction = calibration_report.get("direction", {}) if isinstance(calibration_report.get("direction", {}), dict) else {}
    report_ex = report_direction.get("explainability", {}) if isinstance(report_direction.get("explainability", {}), dict) else {}
    direction_ex = direction_info.get("explainability", {}) if isinstance(direction_info.get("explainability", {}), dict) else {}

    report_gate_funnel = report_ex.get("gate_funnel", {}) if isinstance(report_ex.get("gate_funnel", {}), dict) else {}
    direction_gate_funnel = direction_ex.get("gate_funnel", {}) if isinstance(direction_ex.get("gate_funnel", {}), dict) else {}

    if report_gate_funnel and direction_gate_funnel and report_gate_funnel != direction_gate_funnel:
        print("[WARN] calibration_report.direction.explainability.gate_funnel differs from direction_info.explainability.gate_funnel; ml_ready export will use calibration_report as canonical source.")

    return report_ex or direction_ex



def _build_ml_ready_exports(
    calibration_report: Dict[str, Any],
    config: Dict[str, Any],
    direction_info: Dict[str, Any],
    input_video_path: Path,
) -> Dict[str, Any]:
    run_id = str(calibration_report.get("run_id", "unknown_run"))
    video_id = input_video_path.stem
    ex = _resolve_export_direction_explainability(calibration_report, direction_info)
    manual_review = calibration_report.get("manual_review", {}) if isinstance(calibration_report.get("manual_review", {}), dict) else {}
    gate_funnel = ex.get("gate_funnel", {}) if isinstance(ex.get("gate_funnel", {}), dict) else {}

    auto_status = str(direction_info.get("auto_inference", {}).get("status", ""))
    direction_reliable = bool(direction_info.get("reliable", False))
    voting_tracks = int(ex.get("voting_tracks", 0) or 0)
    top_k_limit = int(ex.get("top_k_limit", 0) or 0)
    min_voting_tracks = int(
        ex.get("min_voting_tracks")
        or config.get("calibration", {}).get("direction_auto", {}).get("min_voting_tracks", 2)
    )

    if (not direction_reliable) and (auto_status == "failed_insufficient_candidates" or voting_tracks < min_voting_tracks):
        direction_risk_semantic_class = "evidence_insufficient"
        label_train_eligibility = "weak_only"
        weak_label_direction_quality = "fallback_insufficient"
    elif any(r.get("code") == "DIRECTION_CONFIDENT_BUT_EVENT_PROFILE_CONFLICT" for r in manual_review.get("reasons", [])):
        direction_risk_semantic_class = "confident_conflict"
        label_train_eligibility = "eligible"
        weak_label_direction_quality = "sufficient_auto"
    else:
        direction_risk_semantic_class = "none"
        label_train_eligibility = "eligible" if direction_reliable else "weak_only"
        weak_label_direction_quality = "sufficient_auto" if direction_reliable else "fallback_other"

    selected_window_ids = {int(x) for x in ex.get("selected_event_window_ids", []) if isinstance(x, (int, float))}
    filtered_reason_by_track = {
        int(x.get("track_id")): str(x.get("reason"))
        for x in ex.get("excluded_tracks", [])
        if isinstance(x, dict) and x.get("track_id") is not None
    }

    track_rows: List[Dict[str, Any]] = []
    for item in ex.get("top_k_selected", []):
        if not isinstance(item, dict):
            continue
        feats = item.get("features", {}) if isinstance(item.get("features", {}), dict) else {}
        tid = int(item.get("track_id", -1))
        row = {
            "meta_schema_version": "1.0",
            "run_id": run_id,
            "video_id": video_id,
            "window_id": int(item.get("window_id", -1)),
            "track_id": tid,
            "main_subject_score": float(item.get("main_subject_score", 0.0)),
            "delta_x": float(item.get("delta_x", 0.0)),
            "median_area": float(feats.get("median_area", 0.0)),
            "median_height": float(feats.get("median_height", 0.0)),
            "move_abs_px": float(feats.get("move_abs_px", 0.0)),
            "crossing_quality": float(feats.get("crossing_quality", 0.0)),
            "sparse_score": float(feats.get("sparse_score", 0.0)),
            "continuity_score": float(feats.get("continuity_score", 0.0)),
            "core_distance_ratio": float(feats.get("core_distance_ratio", 0.0)),
            "size_score": float(feats.get("size_score", 0.0)),
            "motion_score": float(feats.get("motion_score", 0.0)),
            "window_rank_score": float(item.get("window_rank_score", 0.0)),
            "subjectness_boost": float(item.get("subjectness_boost", 0.0)),
            "base_combined_score": float(item.get("base_combined_score", 0.0)),
            "combined_rank_score": float(item.get("combined_rank_score", 0.0)),
            "is_topk_selected": True,
            "weak_label_track_role": "main_candidate",
            "filter_reason": None,
            "review_status": "unreviewed",
            "reviewer_id": None,
            "reviewed_at": None,
            "review_note": None,
        }
        if tid in filtered_reason_by_track:
            row["weak_label_track_role"] = "filtered_out"
            row["filter_reason"] = filtered_reason_by_track[tid]
        track_rows.append(row)

    window_rows: List[Dict[str, Any]] = []
    for item in ex.get("event_windows", []):
        if not isinstance(item, dict):
            continue
        wid = int(item.get("window_id", -1))
        crowded = bool(item.get("crowded_background_group", False))
        is_selected = wid in selected_window_ids
        weak_role = "main_event_window" if is_selected else ("background_window" if crowded or bool(item.get("suppress_for_selection", False)) else "neutral_window")
        window_rows.append({
            "meta_schema_version": "1.0",
            "run_id": run_id,
            "video_id": video_id,
            "window_id": wid,
            "candidate_count": int(item.get("candidate_count", 0)),
            "top1_track_id": int(item.get("top1_track_id", 0)),
            "top1_main_subject_score": float(item.get("top1_main_subject_score", 0.0)),
            "top2_main_subject_score": float(item.get("top2_main_subject_score", 0.0)),
            "dominance_score": float(item.get("dominance_score", 0.0)),
            "event_likeness_score": float(item.get("event_likeness_score", 0.0)),
            "window_rank_score": float(item.get("window_rank_score", item.get("event_likeness_score", 0.0))),
            "background_context_score": float(item.get("background_context_score", 0.0)),
            "suppression_ratio": float(item.get("suppression_ratio", 0.0)),
            "subjectness_boost": float(item.get("subjectness_boost", 0.0)),
            "small_target_ratio": float(item.get("small_target_ratio", 0.0)),
            "prefilter_context_tracks": int(item.get("prefilter_context_tracks", 0)),
            "quality_filtered_tracks": int(item.get("quality_filtered_tracks", 0)),
            "same_direction_group_count": int(item.get("same_direction_group_count", 0)),
            "similar_scale_group_count": int(item.get("similar_scale_group_count", 0)),
            "top1_size_rank_proxy": float(item.get("top1_size_rank_proxy", 0.0)),
            "crowded_background_group": crowded,
            "suppress_for_selection": bool(item.get("suppress_for_selection", False)),
            "is_selected_event_window": is_selected,
            "weak_label_window_role": weak_role,
            "review_status": "unreviewed",
            "reviewer_id": None,
            "reviewed_at": None,
            "review_note": None,
        })

    video_row = {
        "meta_schema_version": "1.0",
        "run_id": run_id,
        "video_id": video_id,
        "direction_reliable": direction_reliable,
        "auto_inference_status": auto_status,
        "direction_final_value": str(direction_info.get("final", {}).get("value", "left_to_right")),
        "direction_final_source": str(direction_info.get("final", {}).get("source", "default_direction")),
        "final_confidence": float(ex.get("final_confidence", 0.0)),
        "voting_tracks": voting_tracks,
        "top_k_limit": top_k_limit,
        "gate_funnel": gate_funnel,
        "manual_review_recommended": bool(manual_review.get("recommended", False)),
        "manual_review_priority": str(manual_review.get("priority", "low")),
        "manual_reason_codes": [str(x.get("code")) for x in manual_review.get("reasons", []) if isinstance(x, dict) and x.get("code")],
        "direction_risk_semantic_class": direction_risk_semantic_class,
        "label_train_eligibility": label_train_eligibility,
        "weak_label_direction_quality": weak_label_direction_quality,
        "review_status": "unreviewed",
        "reviewer_id": None,
        "reviewed_at": None,
        "review_note": None,
    }

    return {
        "track_rows": track_rows,
        "window_rows": window_rows,
        "video_rows": [video_row],
    }


def _export_ml_ready_artifacts(config: Dict[str, Any], calibration_report: Dict[str, Any], direction_info: Dict[str, Any], input_video_path: Path) -> None:
    try:
        output_root = Path(config.get("output", {}).get("root_dir", "output"))
        ml_ready_dir = output_root / "ml_ready"
        exports = _build_ml_ready_exports(calibration_report, config, direction_info, input_video_path)
        track_path = ml_ready_dir / "track_samples.jsonl"
        window_path = ml_ready_dir / "window_samples.jsonl"
        video_path = ml_ready_dir / "video_samples.jsonl"
        _safe_write_jsonl(exports["track_rows"], track_path)
        _safe_write_jsonl(exports["window_rows"], window_path)
        _safe_write_jsonl(exports["video_rows"], video_path)
        manifest = {
            "meta_schema_version": "1.0",
            "run_id": str(calibration_report.get("run_id", "unknown_run")),
            "exported_at": _utc_now_iso(),
            "export_root": str(ml_ready_dir),
            "files": {
                "track_samples": {"path": "track_samples.jsonl", "rows": len(exports["track_rows"]), "primary_key": ["run_id", "video_id", "window_id", "track_id"]},
                "window_samples": {"path": "window_samples.jsonl", "rows": len(exports["window_rows"]), "primary_key": ["run_id", "video_id", "window_id"]},
                "video_samples": {"path": "video_samples.jsonl", "rows": len(exports["video_rows"]), "primary_key": ["run_id", "video_id"]},
            },
            "label_policy": {
                "weak_label_sources": ["rule_engine", "manual_review"],
                "direction_risk_semantic_class_values": ["none", "confident_conflict", "evidence_insufficient"],
                "label_train_eligibility_values": ["eligible", "weak_only"],
            },
            "compatibility": {
                "non_intrusive_export": True,
                "does_not_modify_calibration_report": True,
                "does_not_modify_priority_trace_schema": True,
            },
        }
        write_json(manifest, ml_ready_dir / "manifest.json")
        print(f"[INFO] ML-ready artifacts exported: {ml_ready_dir}")
    except Exception as exc:
        print(f"[WARN] Failed to export ML-ready artifacts: {exc}")


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

MVP_TRACE_MAX_TRIGGERED_RULES = 5
MVP_TRACE_FIELDS = {
    "trace_version",
    "event_id",
    "timestamp",
    "input_window",
    "source",
    "severity_counts",
    "triggered_rules",
    "primary_rule",
    "final_priority",
    "decision_path",
}


def _build_severity_counts(risk_flags: List[str]) -> Dict[str, int]:
    high_codes = {
        "active_fallback",
        "direction_unreliable",
        "roi_low_confidence_geometry",
        "roi_geometry_anomaly",
        "roi_geometry_invalid",
        "direction_background_biased_event_windows",
        "direction_confident_but_event_profile_conflict",
    }
    medium_codes = {
        "roi_low_score",
        "roi_geometry_too_close_border",
        "direction_topk_weak_foregroundness",
    }

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for code in risk_flags:
        if code in high_codes:
            counts["high"] += 1
        elif code in medium_codes:
            counts["medium"] += 1
        else:
            counts["low"] += 1

    counts["total"] = counts["critical"] + counts["high"] + counts["medium"] + counts["low"] + counts["info"]
    return counts


def _build_priority_decision_trace(event_id: str, input_window: Dict[str, str], risk_flags: List[str]) -> Dict[str, Any]:
    severity_counts = _build_severity_counts(risk_flags)
    hard_high_background = "direction_background_biased_event_windows" in risk_flags
    hard_high_profile_conflict = "direction_confident_but_event_profile_conflict" in risk_flags

    rules = [
        {
            "rule_id": "R_HARD_BACKGROUND_BIASED_WINDOWS",
            "matched": hard_high_background,
            "proposed_priority": "high",
            "explanation": "hard high trigger: direction_background_biased_event_windows is present",
            "decisive_evidence": {"risk_flag": "direction_background_biased_event_windows", "present": hard_high_background},
        },
        {
            "rule_id": "R_HARD_DIRECTION_PROFILE_CONFLICT",
            "matched": hard_high_profile_conflict,
            "proposed_priority": "high",
            "explanation": "hard high trigger: direction_confident_but_event_profile_conflict is present",
            "decisive_evidence": {"risk_flag": "direction_confident_but_event_profile_conflict", "present": hard_high_profile_conflict},
        },
        {
            "rule_id": "R_HIGH_SEVERITY_PRESENT",
            "matched": severity_counts["high"] > 0,
            "proposed_priority": "high",
            "explanation": f"high severity count = {severity_counts['high']} (>0)",
            "decisive_evidence": {"severity_counts.high": severity_counts["high"], "threshold": 1},
        },
        {
            "rule_id": "R_MULTI_SIGNAL_HIGH",
            "matched": severity_counts["total"] >= 2,
            "proposed_priority": "high",
            "explanation": f"total signals = {severity_counts['total']} (>=2)",
            "decisive_evidence": {"severity_counts.total": severity_counts["total"], "threshold": 2},
        },
        {
            "rule_id": "R_MEDIUM_SEVERITY_PRESENT",
            "matched": severity_counts["medium"] > 0,
            "proposed_priority": "medium",
            "explanation": f"medium severity count = {severity_counts['medium']} (>0)",
            "decisive_evidence": {"severity_counts.medium": severity_counts["medium"], "threshold": 1},
        },
        {
            "rule_id": "R_DEFAULT_LOW",
            "matched": severity_counts["total"] == 0,
            "proposed_priority": "low",
            "explanation": "no higher-priority rule matched",
            "decisive_evidence": {"severity_counts.total": severity_counts["total"]},
        },
    ]

    matched_rules = []
    for r in rules:
        if not r["matched"]:
            continue
        matched_rules.append({
            "rule_id": r["rule_id"],
            "matched": True,
            "proposed_priority": r["proposed_priority"],
            "explanation": r["explanation"],
        })
        if len(matched_rules) >= MVP_TRACE_MAX_TRIGGERED_RULES:
            break

    if not matched_rules:
        matched_rules = [{
            "rule_id": "R_DEFAULT_LOW",
            "matched": True,
            "proposed_priority": "low",
            "explanation": "fallback default low",
        }]

    primary_rule_meta = next((r for r in rules if r["rule_id"] == matched_rules[0]["rule_id"]), rules[-1])
    final_priority = matched_rules[0]["proposed_priority"]

    trace = {
        "trace_version": "1.0",
        "event_id": event_id,
        "timestamp": _utc_now_iso(),
        "input_window": {
            "start_ts": input_window["start_ts"],
            "end_ts": input_window["end_ts"],
        },
        "source": {
            "engine": "priority-rule-engine",
            "engine_version": "0.1.0",
        },
        "severity_counts": severity_counts,
        "triggered_rules": matched_rules,
        "primary_rule": {
            "rule_id": matched_rules[0]["rule_id"],
            "decision_basis": f"Matched highest-priority rule: {matched_rules[0]['rule_id']}",
            "decisive_evidence": primary_rule_meta["decisive_evidence"],
        },
        "final_priority": final_priority,
        "decision_path": f"{matched_rules[0]['rule_id']} -> {final_priority}",
    }

    _validate_priority_trace_mvp(trace)
    return trace


def _validate_priority_trace_mvp(trace: Dict[str, Any]) -> None:
    if set(trace.keys()) != MVP_TRACE_FIELDS:
        raise ValueError("priority_decision_trace fields must strictly match MVP 10 fields")

    source = trace.get("source", {})
    if not isinstance(source, dict) or not source.get("engine") or not source.get("engine_version"):
        raise ValueError("priority_decision_trace.source must include engine and engine_version")

    sc = trace.get("severity_counts", {})
    required_sc = ["critical", "high", "medium", "low", "info", "total"]
    if any(k not in sc for k in required_sc):
        raise ValueError("severity_counts missing required keys")
    if sc["total"] != sc["critical"] + sc["high"] + sc["medium"] + sc["low"] + sc["info"]:
        raise ValueError("severity_counts.total mismatch")

    trules = trace.get("triggered_rules", [])
    if not trules or len(trules) > MVP_TRACE_MAX_TRIGGERED_RULES:
        raise ValueError("triggered_rules must contain 1..5 matched rules")
    if any((not isinstance(r, dict)) or (r.get("matched") is not True) for r in trules):
        raise ValueError("triggered_rules must only contain matched rules")

    if trace.get("final_priority") not in {"high", "medium", "low"}:
        raise ValueError("final_priority must be high/medium/low")


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



def _track_motion_summary(history: List[Tuple[int, BBox]], motion_min_frames: int, direction: str, direction_min_progress_px: float) -> Dict[str, float]:
    if len(history) < motion_min_frames:
        return {"enough_history": False, "dx": 0.0, "dy": 0.0, "dist": 0.0, "direction_ok": False}
    window = history[-motion_min_frames:]
    start_cx, start_cy = _bbox_center(window[0][1])
    end_cx, end_cy = _bbox_center(window[-1][1])
    dx = end_cx - start_cx
    dy = end_cy - start_cy
    dist = math.sqrt(dx * dx + dy * dy)
    direction_ok = (direction == "left_to_right" and dx >= direction_min_progress_px) or (direction == "right_to_left" and dx <= -direction_min_progress_px)
    return {"enough_history": True, "dx": dx, "dy": dy, "dist": dist, "direction_ok": direction_ok}



def _select_calibration_candidate_track_ids(tracker: MinimalTracker, direction: str, min_track_frames: int, motion_min_frames: int, motion_min_distance_px: float, direction_min_progress_px: float) -> List[int]:
    candidate_ids: List[int] = []
    for tid, track in tracker.tracks.items():
        if track.confirmed:
            candidate_ids.append(tid)
            continue
        if track.hit_frames < min_track_frames:
            continue
        motion = _track_motion_summary(track.history, motion_min_frames, direction, direction_min_progress_px)
        if not motion["enough_history"]:
            continue
        if motion["dist"] < motion_min_distance_px:
            continue
        if not motion["direction_ok"]:
            continue
        candidate_ids.append(tid)
    return candidate_ids


def _is_corridor_aligned_high_track(med_y: float, active_roi: ROI, min_bottom_ratio: float, core_dist_ratio: float, max_core_distance_ratio: float, near_core_count: int, horiz_axis_score: float) -> bool:
    _, ay, _, ah = active_roi
    if med_y >= ay + ah * min_bottom_ratio:
        return False
    if core_dist_ratio > max_core_distance_ratio:
        return False
    if near_core_count < 2:
        return False
    if horiz_axis_score < 0.35:
        return False
    return True


def _is_strong_subject_high_track(med_y: float, upper_background_threshold_y: float, active_roi: ROI, move_abs: float, min_move_px: float, consistency: float, min_consistency: float, horiz_axis_score: float, dxdy_ratio: float) -> bool:
    _, _, _, ah = active_roi
    if med_y >= upper_background_threshold_y:
        return False
    upper_background_gap_px = upper_background_threshold_y - med_y
    upper_background_risk_ratio = upper_background_gap_px / max(1.0, ah)
    if upper_background_risk_ratio > 0.18:
        return False
    if move_abs < max(min_move_px * 2.0, min_move_px + 40.0):
        return False
    if consistency < max(min_consistency, 0.95):
        return False
    if horiz_axis_score < 0.5:
        return False
    if dxdy_ratio < 1.5:
        return False
    return True


def _classify_active_frame_candidate(raw_roi: ROI, frame_width: int, min_acceptable_narrow_width_ratio: float, max_content_width_ratio: float, center_offset_max_ratio: float, symmetry_tolerance_ratio: float) -> Tuple[str, Dict[str, float]]:
    ratio = raw_roi[2] / max(1, frame_width)
    content_center = raw_roi[0] + raw_roi[2] / 2.0
    center_offset_ratio = abs(content_center - (frame_width / 2.0)) / max(1.0, frame_width)
    left_bar = raw_roi[0]
    right_bar = max(0, frame_width - (raw_roi[0] + raw_roi[2]))
    symmetry_ratio = abs(left_bar - right_bar) / max(1.0, frame_width)
    stats = {
        "center_offset_ratio": round(center_offset_ratio, 4),
        "side_symmetry_ratio": round(symmetry_ratio, 4),
        "final_width_ratio": round(ratio, 4),
    }
    if ratio < min_acceptable_narrow_width_ratio:
        return "too_narrow", stats
    if center_offset_ratio > center_offset_max_ratio:
        return "off_center", stats
    if symmetry_ratio > symmetry_tolerance_ratio:
        return "asymmetric_side_bars", stats
    if ratio > max_content_width_ratio:
        return "near_full_width", stats
    return "ok", stats



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
        geo_status, geo_stats = _classify_active_frame_candidate(
            raw,
            width,
            min_acceptable_narrow_width_ratio,
            max_ratio,
            center_offset_max_ratio,
            symmetry_tolerance_ratio,
        )
        near_full_width_accept = geo_status == "near_full_width" and jitter <= max_jitter and pass_ratio >= 0.3
        if jitter <= max_jitter and pass_ratio >= 0.3 and geo_status in ("ok", "near_full_width"):
            validity = "ok_full_width_stable" if near_full_width_accept else "ok"
            acceptance_mode = "full_width_conditional_accept" if near_full_width_accept else "direct"
            roi_clipped, _ = _clip_roi_to_bounds(raw, width, height)
            if near_full_width_accept:
                warnings.append("accepted near-full-width active-frame candidate because it remained stable, centered, and symmetric across sampled frames")
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
        validity = "failed_geometry_rejected"
        rejected_candidate_info = {
            "reason": geo_status if geo_status not in ("ok", "near_full_width") else ("unstable_jitter" if jitter > max_jitter else "low_valid_frame_ratio"),
            "candidate_roi": _roi_to_cfg(raw),
            "candidate_type": "normal_width",
            "jitter_px": round(jitter, 2),
            "valid_ratio": round(pass_ratio, 3),
            **geo_stats,
        }

    # Priority 2: narrow but stable candidates accepted as success
    if narrow_widths:
        nleft = int(_quantile([float(x) for x in narrow_lefts], left_q, median(narrow_lefts)))
        nright = int(_quantile([float(x) for x in narrow_rights], right_q, median(narrow_rights)))
        nraw = (nleft, 0, max(0, nright - nleft + 1), height)
        detected_candidate_roi = nraw
        njitter = float(max(narrow_widths) - min(narrow_widths)) if narrow_widths else 0.0
        nratio = len(narrow_widths) / max(1, sampled)
        geo_status, geo_stats = _classify_active_frame_candidate(
            nraw,
            width,
            min_acceptable_narrow_width_ratio,
            max_ratio,
            center_offset_max_ratio,
            symmetry_tolerance_ratio,
        )
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
        validity = "failed_geometry_rejected"
        rejected_candidate_info = {
            "reason": geo_status if geo_status != "ok" else "narrow_candidate_not_stable_enough",
            "candidate_roi": _roi_to_cfg(nraw),
            "candidate_type": "narrow_stable",
            "jitter_px": round(njitter, 2),
            "valid_frames": len(narrow_widths),
            **geo_stats,
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
                "total_tracks": len(track_features),
                "filtered_out_tracks": 0,
                "prefilter_pass_tracks": 0,
                "top_k_limit": 0,
                "voting_tracks": 0,
                "left_to_right_score": 0.0,
                "right_to_left_score": 0.0,
                "final_confidence": 1.0,
                "top_k_selected": [],
                "excluded_tracks": [],
                "excluded_reason_counts": {},
                "main_subject_profile_confidence": 1.0,
                "event_windows": [],
                "selected_event_window_ids": [],
                "crowded_background_group_windows": [],
                "crowded_suppressed_candidate_count": 0,
            },
        }

    active_unstable = active_result.get("fallback_used", False) or active_result.get("validity") not in ("ok", "ok_narrow_stable", "ok_full_width_stable")
    ax, ay, aw, ah = active_roi
    cx, _, cw, _ = core_roi
    total_tracks = len(track_features)

    min_track_points = int(dir_cfg.get("min_track_points", 4))
    min_move_px = float(dir_cfg.get("min_move_px", 52))
    min_area = float(dir_cfg.get("min_bbox_area_px", 2000))
    min_height = float(dir_cfg.get("min_bbox_height_px", 58))
    min_bottom_ratio = float(dir_cfg.get("min_bottom_region_ratio", 0.58))
    min_crossing_bottom_ratio = float(dir_cfg.get("min_crossing_bottom_ratio", 0.60))
    max_core_distance_ratio = float(dir_cfg.get("max_core_distance_ratio", 0.28))
    min_consistency = float(dir_cfg.get("min_direction_consistency", 0.66))
    min_voting_tracks = int(dir_cfg.get("min_voting_tracks", 2))
    min_final_confidence = float(dir_cfg.get("min_final_confidence", 0.68))

    top_k = int(dir_cfg.get("top_k", 4))
    min_main_subject_score = float(dir_cfg.get("min_main_subject_score", 0.45))

    w_size = float(dir_cfg.get("score_weight_size", 0.32))
    w_motion = float(dir_cfg.get("score_weight_motion", 0.30))
    w_cross = float(dir_cfg.get("score_weight_crossing", 0.24))
    w_sparse = float(dir_cfg.get("score_weight_sparse", 0.08))
    w_cont = float(dir_cfg.get("score_weight_continuity", 0.06))

    # window settings
    window_size_frames = int(dir_cfg.get("window_size_frames", 45))
    select_top_windows = int(dir_cfg.get("select_top_windows", 2))
    crowded_min_candidates = int(dir_cfg.get("crowded_min_candidates", 4))
    crowded_small_ratio = float(dir_cfg.get("crowded_small_ratio", 0.65))
    dominance_min_top1_score = float(dir_cfg.get("dominance_min_top1_score", 0.58))

    reason_counts: Dict[str, int] = {}
    excluded_tracks: List[Dict[str, Any]] = []
    prefilter_diagnostics: List[Dict[str, Any]] = []
    excluded_quality_by_window: Dict[int, int] = {}
    prefilter_candidates: List[Dict[str, Any]] = []

    # determine coarse crowd density map from centers
    center_bins: Dict[int, int] = {}
    for feat in track_features.values():
        pts = feat.get("points", [])
        if not pts:
            continue
        midx = int(median([p[0] for p in pts]) / max(1.0, aw * 0.12))
        center_bins[midx] = center_bins.get(midx, 0) + 1

    def _exclude(tid: int, reason: str, window_id: Optional[int] = None, diagnostic: Optional[Dict[str, Any]] = None) -> None:
        normalized_window_id = int(window_id) if window_id is not None and int(window_id) >= 0 else None
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        excluded_tracks.append({
            "track_id": tid,
            "reason": reason,
            "window_id": normalized_window_id,
            "filter_stage": "quality_gate",
        })
        if diagnostic is not None:
            diag = {
                "track_id": tid,
                "reason": reason,
                "window_id": normalized_window_id,
            }
            diag.update(diagnostic)
            prefilter_diagnostics.append(diag)
        if normalized_window_id is not None:
            excluded_quality_by_window[normalized_window_id] = excluded_quality_by_window.get(normalized_window_id, 0) + 1

    for tid, feat in track_features.items():
        pts = feat.get("points", [])
        frames = feat.get("frame_indices", [])
        window_hint = int(median(frames) // max(1, window_size_frames)) if frames else -1
        if len(pts) < min_track_points:
            _exclude(tid, "short_track", window_hint)
            continue
        if len(frames) < min_track_points:
            _exclude(tid, "short_track_frames", window_hint)
            continue

        areas = feat.get("areas", [])
        heights = feat.get("heights", [])
        if not areas or not heights:
            _exclude(tid, "missing_geometry", window_hint)
            continue

        med_area = float(median(areas))
        med_h = float(median(heights))
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        med_y = float(median(ys))
        delta_x = float(pts[-1][0] - pts[0][0])
        delta_y = float(pts[-1][1] - pts[0][1])
        move_abs = abs(delta_x)

        deltas = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
        if not deltas:
            _exclude(tid, "no_motion_series", window_hint)
            continue
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        consistency = max(pos, neg) / max(1, pos + neg)

        core_dist_ratio = abs(float(median(xs)) - (cx + cw / 2.0)) / max(1.0, aw)

        # hard filters
        if med_area < min_area:
            _exclude(tid, "small_bbox", window_hint)
            continue
        if med_h < min_height:
            _exclude(tid, "small_height", window_hint)
            continue
        if move_abs < min_move_px:
            _exclude(tid, "low_horizontal_motion", window_hint)
            continue
        if consistency < min_consistency:
            _exclude(tid, "low_direction_consistency", window_hint)
            continue

        # horizontal crossing quality (core proximity is embedded, not standalone)
        dxdy_ratio = abs(delta_x) / max(1.0, abs(delta_y))
        horiz_axis_score = min(1.0, dxdy_ratio / 3.0)
        near_core_points = [p for p in pts if abs(p[0] - (cx + cw / 2.0)) <= max(20.0, cw * 0.8)]
        near_core_count = len(near_core_points)
        corridor_aligned_high_track = _is_corridor_aligned_high_track(
            med_y,
            active_roi,
            min_bottom_ratio,
            core_dist_ratio,
            max_core_distance_ratio,
            near_core_count,
            horiz_axis_score,
        )
        upper_background_threshold_y = ay + ah * min_bottom_ratio
        crossing_lower_band_threshold_y = ay + ah * min_crossing_bottom_ratio
        strong_subject_high_track_override = _is_strong_subject_high_track(
            med_y,
            upper_background_threshold_y,
            active_roi,
            move_abs,
            min_move_px,
            consistency,
            min_consistency,
            horiz_axis_score,
            dxdy_ratio,
        )
        high_track_override = corridor_aligned_high_track or strong_subject_high_track_override
        near_core_dx_quality = min(1.0, move_abs / max(1.0, min_move_px * 1.5)) if near_core_count > 0 else 0.0
        crossing_bottom_gate = 1.0 if med_y >= crossing_lower_band_threshold_y else (0.78 if strong_subject_high_track_override else (0.72 if corridor_aligned_high_track else 0.0))
        upper_background_gap_px = max(0.0, upper_background_threshold_y - med_y)
        upper_background_risk_ratio = upper_background_gap_px / max(1.0, ah)
        prefilter_diag = {
            "med_y": round(med_y, 3),
            "upper_background_threshold_y": round(upper_background_threshold_y, 3),
            "crossing_lower_band_threshold_y": round(crossing_lower_band_threshold_y, 3),
            "core_dist_ratio": round(core_dist_ratio, 4),
            "near_core_count": int(near_core_count),
            "horiz_axis_score": round(horiz_axis_score, 4),
            "dxdy_ratio": round(dxdy_ratio, 4),
            "crossing_bottom_gate": round(crossing_bottom_gate, 3),
            "corridor_aligned_high_track": bool(corridor_aligned_high_track),
            "strong_subject_high_track_override": bool(strong_subject_high_track_override),
            "upper_background_gap_px": round(upper_background_gap_px, 3),
            "upper_background_risk_ratio": round(upper_background_risk_ratio, 4),
            "move_abs": round(move_abs, 3),
            "consistency": round(consistency, 4),
        }
        if med_y < upper_background_threshold_y and not high_track_override:
            _exclude(tid, "upper_background", window_hint, diagnostic=prefilter_diag)
            continue
        if crossing_bottom_gate == 0.0:
            _exclude(tid, "crossing_not_in_lower_band", window_hint, diagnostic=prefilter_diag)
            continue
        corridor_component = max(0.0, 1.0 - min(1.0, core_dist_ratio / max(1e-6, max_core_distance_ratio)))
        crossing_quality = ((0.40 * horiz_axis_score) + (0.35 * near_core_dx_quality) + (0.25 * corridor_component)) * crossing_bottom_gate
        if crossing_quality < float(dir_cfg.get("min_crossing_quality", 0.45)):
            diag_low_crossing = dict(prefilter_diag)
            diag_low_crossing["crossing_quality"] = round(crossing_quality, 4)
            _exclude(tid, "low_crossing_quality", window_hint, diagnostic=diag_low_crossing)
            continue

        s_size = min(1.0, med_area / max(1.0, min_area * 2.0))
        s_motion = min(1.0, move_abs / max(1.0, min_move_px * 2.0))

        midx = int(median(xs) / max(1.0, aw * 0.12))
        local_crowd = center_bins.get(midx, 1)
        s_sparse = max(0.0, 1.0 - min(1.0, (local_crowd - 1) / 4.0))

        continuity = len(pts) / max(1.0, min_track_points * 2.0)
        s_cont = min(1.0, continuity)

        main_subject_score = (w_size * s_size) + (w_motion * s_motion) + (w_cross * crossing_quality) + (w_sparse * s_sparse) + (w_cont * s_cont)

        if main_subject_score < min_main_subject_score:
            _exclude(tid, "below_main_subject_score", window_hint)
            continue

        median_frame = int(median(frames))
        window_id = int(median_frame // max(1, window_size_frames))

        prefilter_candidates.append({
            "track_id": tid,
            "window_id": window_id,
            "delta_x": round(delta_x, 3),
            "main_subject_score": round(main_subject_score, 4),
            "features": {
                "size_score": round(s_size, 3),
                "motion_score": round(s_motion, 3),
                "crossing_quality": round(crossing_quality, 3),
                "sparse_score": round(s_sparse, 3),
                "continuity_score": round(s_cont, 3),
                "median_area": round(med_area, 1),
                "median_height": round(med_h, 1),
                "median_center_y": round(med_y, 1),
                "core_distance_ratio": round(core_dist_ratio, 4),
                "move_abs_px": round(move_abs, 2),
                "near_core_point_count": near_core_count,
                "corridor_aligned_high_track": corridor_aligned_high_track,
                "strong_subject_high_track_override": strong_subject_high_track_override,
                "crossing_bottom_gate": round(crossing_bottom_gate, 3),
                "dxdy_ratio": round(dxdy_ratio, 3),
            },
        })

    # event window scoring with context-aware crowd/background signals
    windows: Dict[int, List[Dict[str, Any]]] = {}
    for c in prefilter_candidates:
        windows.setdefault(int(c["window_id"]), []).append(c)

    event_windows: List[Dict[str, Any]] = []
    crowded_window_ids: List[int] = []
    crowded_suppressed_candidate_count = 0

    # internal-only split counters (quality gating vs background-context suppression)
    quality_filtered_track_count = len(excluded_tracks)
    background_context_suppressed_track_count = 0

    for wid, items in windows.items():
        items_sorted = sorted(items, key=lambda x: x["main_subject_score"], reverse=True)
        top1 = items_sorted[0]
        top2 = items_sorted[1] if len(items_sorted) > 1 else None

        top1_score = float(top1["main_subject_score"])
        top2_score = float(top2["main_subject_score"]) if top2 else 0.0
        gap = max(0.0, top1_score - top2_score)
        dominance = min(1.0, 0.55 * gap / max(1e-6, 1.0) + 0.45 * top1_score)
        if top1_score < dominance_min_top1_score:
            dominance *= 0.6

        candidate_count = len(items_sorted)
        smallish = sum(1 for x in items_sorted if float(x["features"].get("size_score", 0.0)) < 0.55)
        small_ratio = smallish / max(1, candidate_count)

        # context metrics: do not rely only on final candidate_count
        quality_filtered_same_window = int(excluded_quality_by_window.get(int(wid), 0))
        neighbor_prefilter_tracks = sum(len(windows.get(int(nwid), [])) for nwid in [wid - 1, wid + 1])
        neighbor_quality_filtered_tracks = sum(int(excluded_quality_by_window.get(int(nwid), 0)) for nwid in [wid - 1, wid + 1])
        prefilter_context_tracks = candidate_count + quality_filtered_same_window + neighbor_prefilter_tracks + neighbor_quality_filtered_tracks

        top1_delta = float(top1.get("delta_x", 0.0))
        top1_sign = 1 if top1_delta >= 0 else -1
        top1_size_score = float(top1.get("features", {}).get("size_score", 0.0))

        same_direction_group_count = 0
        similar_scale_group_count = 0
        for item in items_sorted:
            dx = float(item.get("delta_x", 0.0))
            sign = 1 if dx >= 0 else -1
            if sign == top1_sign:
                same_direction_group_count += 1
            size_score = float(item.get("features", {}).get("size_score", 0.0))
            if abs(size_score - top1_size_score) <= 0.18:
                similar_scale_group_count += 1

        areas_sorted = sorted(float(x.get("features", {}).get("median_area", 0.0)) for x in items_sorted)
        top1_area = float(top1.get("features", {}).get("median_area", 0.0))
        rank_desc = 1 + sum(1 for a in areas_sorted if a > top1_area)
        top1_size_rank_proxy = 1.0 if candidate_count <= 1 else max(0.0, 1.0 - ((rank_desc - 1) / max(1, candidate_count - 1)))

        suppression_pressure = max(0, prefilter_context_tracks - candidate_count)
        suppression_ratio = suppression_pressure / max(1.0, float(prefilter_context_tracks))
        crowd_density_score = min(1.0, max(0.0, (prefilter_context_tracks - 1) / 6.0))
        suppression_score = min(1.0, suppression_pressure / 4.0)
        group_flow_score = min(1.0, max(0.0, (same_direction_group_count - 1) / 3.0))
        foreground_penalty = 1.0 - top1_size_rank_proxy
        background_context_score = (
            0.32 * crowd_density_score
            + 0.28 * suppression_score
            + 0.16 * min(1.0, small_ratio)
            + 0.14 * group_flow_score
            + 0.10 * foreground_penalty
        )

        crowded = (
            background_context_score >= 0.58
            or (prefilter_context_tracks >= crowded_min_candidates and small_ratio >= crowded_small_ratio)
            or (suppression_pressure >= 3 and same_direction_group_count >= 2)
        )

        crowd_suppressed_tracks = max(0, candidate_count - 1) if crowded else 0
        if crowded:
            crowded_window_ids.append(wid)
            crowded_suppressed_candidate_count += crowd_suppressed_tracks
            background_context_suppressed_track_count += crowd_suppressed_tracks

        isolation = max(0.0, 1.0 - min(1.0, (candidate_count - 1) / 4.0))
        window_crossing = sum(float(x["features"].get("crossing_quality", 0.0)) for x in items_sorted[:2]) / max(1, min(2, len(items_sorted)))
        window_motion = sum(float(x["features"].get("motion_score", 0.0)) for x in items_sorted[:2]) / max(1, min(2, len(items_sorted)))

        event_like_raw = (0.40 * dominance) + (0.25 * isolation) + (0.20 * window_crossing) + (0.15 * window_motion)
        if crowded:
            event_like_raw *= 0.45

        subjectness_boost = max(
            0.55,
            min(
                1.15,
                (0.50 * top1_size_rank_proxy)
                + (0.30 * min(1.0, window_crossing))
                + (0.20 * min(1.0, top1_score)),
            ),
        )
        window_rank_score = max(
            0.02,
            event_like_raw
            * (1.0 - 0.65 * min(1.0, background_context_score))
            * (1.0 - 0.35 * min(1.0, suppression_ratio))
            * (0.90 + 0.10 * subjectness_boost),
        )

        suppress_for_selection = (
            background_context_score >= 0.66
            or suppression_ratio >= 0.62
            or (crowded and background_context_score >= 0.58 and top1_size_rank_proxy < 0.72)
        )

        event_windows.append({
            "window_id": wid,
            "candidate_count": candidate_count,
            "top1_track_id": int(top1["track_id"]),
            "top1_main_subject_score": round(top1_score, 4),
            "top2_main_subject_score": round(top2_score, 4),
            "dominance_score": round(dominance, 4),
            "event_likeness_score": round(event_like_raw, 4),
            "window_rank_score": round(window_rank_score, 4),
            "crowded_background_group": crowded,
            "small_target_ratio": round(small_ratio, 3),
            "isolation_score": round(isolation, 3),
            "window_crossing_score": round(window_crossing, 3),
            "window_motion_score": round(window_motion, 3),
            "candidate_track_ids": [int(x["track_id"]) for x in items_sorted],
            "prefilter_context_tracks": int(prefilter_context_tracks),
            "quality_filtered_tracks": int(quality_filtered_same_window + neighbor_quality_filtered_tracks),
            "neighbor_prefilter_tracks": int(neighbor_prefilter_tracks),
            "same_direction_group_count": int(same_direction_group_count),
            "similar_scale_group_count": int(similar_scale_group_count),
            "top1_size_rank_proxy": round(top1_size_rank_proxy, 3),
            "background_context_score": round(background_context_score, 4),
            "suppression_ratio": round(suppression_ratio, 4),
            "subjectness_boost": round(subjectness_boost, 4),
            "crowd_suppressed_tracks": int(crowd_suppressed_tracks),
            "suppress_for_selection": bool(suppress_for_selection),
        })

    # choose top event windows: prefer non-suppressed windows first, and refill at most 1 suppressed window
    normal_windows = [w for w in event_windows if not bool(w.get("suppress_for_selection", False))]
    suppressed_windows = [w for w in event_windows if bool(w.get("suppress_for_selection", False))]
    normal_windows.sort(key=lambda w: float(w.get("window_rank_score", w.get("event_likeness_score", 0.0))), reverse=True)
    suppressed_windows.sort(key=lambda w: float(w.get("window_rank_score", w.get("event_likeness_score", 0.0))), reverse=True)

    selected_windows = normal_windows[: max(1, select_top_windows)]
    need_refill = max(0, max(1, select_top_windows) - len(selected_windows))
    if need_refill > 0 and suppressed_windows:
        selected_windows.extend(suppressed_windows[:1])
    selected_windows.sort(key=lambda w: float(w.get("window_rank_score", w.get("event_likeness_score", 0.0))), reverse=True)
    selected_windows = selected_windows[: max(1, select_top_windows)]
    selected_window_ids = [int(w["window_id"]) for w in selected_windows]

    # apply selected-window ranking and select global top-k
    window_like_map = {int(w["window_id"]): float(w.get("window_rank_score", w.get("event_likeness_score", 0.0))) for w in selected_windows}
    selected_candidates = [c for c in prefilter_candidates if int(c["window_id"]) in window_like_map]
    for c in selected_candidates:
        win_score = float(window_like_map[int(c["window_id"])] )
        feats = c.get("features", {})
        track_subjectness = max(
            0.60,
            min(
                1.20,
                (0.35 * float(feats.get("crossing_quality", 0.0)))
                + (0.25 * float(feats.get("size_score", 0.0)))
                + (0.20 * float(feats.get("continuity_score", 0.0)))
                + (0.20 * max(0.0, 1.0 - float(feats.get("core_distance_ratio", 1.0)))),
            ),
        )
        c["window_rank_score"] = round(win_score, 4)
        c["subjectness_boost"] = round(track_subjectness, 4)
        c["base_combined_score"] = round(float(c["main_subject_score"]) * win_score * track_subjectness, 4)

    selected_candidates.sort(key=lambda x: x.get("base_combined_score", 0.0), reverse=True)

    # lightweight anti-background direction consistency (no assumed target direction)
    seed_n = max(2, min(len(selected_candidates), max(2, top_k)))
    seed = selected_candidates[:seed_n]
    seed_flow = sum(float(c.get("base_combined_score", 0.0)) * float(c.get("delta_x", 0.0)) for c in seed)
    consensus_sign = 1 if seed_flow > 0 else (-1 if seed_flow < 0 else 0)

    for c in selected_candidates:
        final_score = float(c.get("base_combined_score", 0.0))
        if consensus_sign != 0:
            dx = float(c.get("delta_x", 0.0))
            feats = c.get("features", {})
            weak_opposite = (
                (1 if dx >= 0 else -1) != consensus_sign
                and abs(dx) < (min_move_px * 1.25)
                and float(feats.get("crossing_quality", 0.0)) < 0.62
                and float(feats.get("size_score", 0.0)) < 0.65
            )
            if weak_opposite:
                final_score *= 0.82
        c["combined_rank_score"] = round(final_score, 4)

    selected_candidates.sort(key=lambda x: x["combined_rank_score"], reverse=True)
    voting_candidates = selected_candidates[: max(0, top_k)]

    l2r_score = sum(float(c["combined_rank_score"]) * abs(float(c["delta_x"])) for c in voting_candidates if float(c["delta_x"]) > 0)
    r2l_score = sum(float(c["combined_rank_score"]) * abs(float(c["delta_x"])) for c in voting_candidates if float(c["delta_x"]) < 0)
    total_score = l2r_score + r2l_score
    confidence = max(l2r_score, r2l_score) / max(1e-6, total_score) if total_score > 0 else 0.0

    if voting_candidates:
        profile_conf = sum(float(c["main_subject_score"]) for c in voting_candidates) / max(1, len(voting_candidates))
    else:
        profile_conf = 0.0

    gate_funnel = {
        "total_tracks": total_tracks,
        "after_length": total_tracks - reason_counts.get("short_track", 0) - reason_counts.get("short_track_frames", 0),
        "after_size": total_tracks - reason_counts.get("short_track", 0) - reason_counts.get("short_track_frames", 0) - reason_counts.get("small_bbox", 0) - reason_counts.get("small_height", 0),
        "after_motion": total_tracks - reason_counts.get("short_track", 0) - reason_counts.get("short_track_frames", 0) - reason_counts.get("small_bbox", 0) - reason_counts.get("small_height", 0) - reason_counts.get("low_horizontal_motion", 0),
        "after_consistency": total_tracks - reason_counts.get("short_track", 0) - reason_counts.get("short_track_frames", 0) - reason_counts.get("small_bbox", 0) - reason_counts.get("small_height", 0) - reason_counts.get("low_horizontal_motion", 0) - reason_counts.get("low_direction_consistency", 0),
        "after_crossing_quality": len(prefilter_candidates),
        "voting_tracks": len(voting_candidates),
        "top_excluded_reasons": sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:6],
        "thresholds_applied": {
            "min_track_points": min_track_points,
            "min_move_px": min_move_px,
            "min_direction_consistency": min_consistency,
            "min_bbox_area_px": min_area,
        },
    }

    reliable = (not active_unstable) and len(voting_candidates) >= min_voting_tracks and total_score > 0 and confidence >= min_final_confidence
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
                "filtered_out_tracks": len(excluded_tracks),
                "prefilter_pass_tracks": len(prefilter_candidates),
                "top_k_limit": top_k,
                "voting_tracks": len(voting_candidates),
                "left_to_right_score": round(l2r_score, 3),
                "right_to_left_score": round(r2l_score, 3),
                "final_confidence": round(confidence, 3),
                "top_k_selected": voting_candidates,
                "excluded_tracks": excluded_tracks,
                "prefilter_diagnostics": prefilter_diagnostics,
                "excluded_reason_counts": reason_counts,
                "gate_funnel": gate_funnel,
                "main_subject_profile_confidence": round(profile_conf, 3),
                "event_windows": event_windows,
                "selected_event_window_ids": selected_window_ids,
                "crowded_background_group_windows": crowded_window_ids,
                "crowded_suppressed_candidate_count": crowded_suppressed_candidate_count,
                "quality_filtered_track_count": quality_filtered_track_count,
                "background_context_suppressed_track_count": background_context_suppressed_track_count,
            },
        }

    status = "failed_insufficient_candidates" if len(voting_candidates) < min_voting_tracks else ("unstable" if active_unstable else "failed")
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
            "filtered_out_tracks": len(excluded_tracks),
            "prefilter_pass_tracks": len(prefilter_candidates),
            "top_k_limit": top_k,
            "voting_tracks": len(voting_candidates),
            "left_to_right_score": round(l2r_score, 3),
            "right_to_left_score": round(r2l_score, 3),
            "final_confidence": round(min(confidence, 0.55) if active_unstable else confidence, 3),
            "top_k_selected": voting_candidates,
            "excluded_tracks": excluded_tracks,
            "prefilter_diagnostics": prefilter_diagnostics,
            "excluded_reason_counts": reason_counts,
            "gate_funnel": gate_funnel,
            "main_subject_profile_confidence": round(profile_conf, 3),
            "event_windows": event_windows,
            "selected_event_window_ids": selected_window_ids,
            "crowded_background_group_windows": crowded_window_ids,
            "crowded_suppressed_candidate_count": crowded_suppressed_candidate_count,
            "quality_filtered_track_count": quality_filtered_track_count,
            "background_context_suppressed_track_count": background_context_suppressed_track_count,
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
    frame_step = 1
    sample_seconds = int(config.get("calibration", {}).get("roi_suggestion", {}).get("sample_seconds", 20))
    calib_frames_limit = int(max(1, video_info.get("fps", 30.0) * sample_seconds))

    direction_seed = str(config.get("event", {}).get("direction", "left_to_right"))
    calib_min_track_frames = min(int(tracking_cfg.get("min_track_frames", 6)), 4)
    calib_motion_min_frames = min(int(tracking_cfg.get("min_motion_frames", 3)), 2)
    calib_motion_min_distance_px = min(float(tracking_cfg.get("min_motion_distance_px", 10)), 8.0)
    calib_direction_min_progress_px = min(float(tracking_cfg.get("direction_min_progress_px", 5)), 3.0)
    calib_tracking_cfg = dict(tracking_cfg)
    calib_tracking_cfg["use_tracking_roi"] = False
    tracker_calib = MinimalTracker(
        iou_threshold=float(tracking_cfg.get("iou_threshold", 0.25)),
        max_center_distance_px=max(float(tracking_cfg.get("max_center_distance_px", 80)), 110.0),
        max_lost_frames=max(int(tracking_cfg.get("max_lost_frames", 8)), 10),
        min_track_frames=calib_min_track_frames,
        motion_min_frames=calib_motion_min_frames,
        motion_min_distance_px=calib_motion_min_distance_px,
        direction=direction_seed,
        direction_min_progress_px=calib_direction_min_progress_px,
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
        filtered = _filter_detections_for_tracking(dets, calib_tracking_cfg, active_frame_roi, tracking_roi)
        assigns = tracker_calib.update(filtered, fidx)
        candidate_track_ids = set(_select_calibration_candidate_track_ids(
            tracker_calib,
            direction_seed,
            calib_min_track_frames,
            calib_motion_min_frames,
            calib_motion_min_distance_px,
            calib_direction_min_progress_px,
        ))
        candidate_assignments = [a for a in assigns if a.track_id in candidate_track_ids]
        frame_feats = _build_track_features(candidate_assignments)
        for tid, feat in frame_feats.items():
            acc = track_features.setdefault(tid, {"points": [], "areas": [], "heights": [], "center_x": [], "center_y": [], "frame_indices": []})
            for k in ["points", "areas", "heights", "center_x", "center_y"]:
                acc[k].extend(feat[k])
            acc["frame_indices"].extend([fidx] * len(feat["points"]))
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

    def _append_risk(flag: str, code: str, message: str) -> None:
        if flag not in risk_flags:
            risk_flags.append(flag)
        manual_reasons.append({"code": code, "message": message})

    if active_result["fallback_used"]:
        _append_risk(
            "active_fallback",
            "ACTIVE_FRAME_FALLBACK",
            f"active_frame_auto fallback used: source={active_result['fallback_source']}, validity={active_result['validity']}",
        )

    ex = direction_info.get("explainability", {})
    direction_reliable = bool(direction_info.get("reliable", False))
    if not direction_reliable:
        _append_risk(
            "direction_unreliable",
            "DIRECTION_UNRELIABLE_OR_FALLBACK",
            (
                "direction not reliable; "
                f"voting_tracks={ex.get('voting_tracks', 0)}, "
                f"l2r={ex.get('left_to_right_score', 0)}, r2l={ex.get('right_to_left_score', 0)}, "
                f"confidence={ex.get('final_confidence', 0)}, final_source={direction_info.get('final', {}).get('source', 'unknown')}"
            ),
        )

    if roi_suggestion.get("mode") != "directional_template":
        _append_risk(
            "roi_low_confidence_geometry",
            "ROI_GENERATED_UNDER_LOW_CONFIDENCE_GEOMETRY",
            f"roi_suggestion mode={roi_suggestion.get('mode')} due to unreliable direction/context",
        )

    if roi_suggestion["geometry_flags"]:
        _append_risk(
            "roi_geometry_anomaly",
            "ROI_GEOMETRY_ANOMALY",
            "roi_geometry anomalies: " + ", ".join(roi_suggestion["geometry_flags"]),
        )

    # ROI invalidity hard check (e.g., w=0/h=0)
    invalid_roi_names: List[str] = []
    roi_items = ["entry_roi", "core_roi", "exit_roi", "tracking_roi"]
    for k in roi_items:
        r = roi_suggestion["result"].get(k, {})
        if int(r.get("w", 0)) <= 0 or int(r.get("h", 0)) <= 0:
            invalid_roi_names.append(k)

    if invalid_roi_names:
        _append_risk(
            "roi_geometry_invalid",
            "ROI_GEOMETRY_INVALID",
            "invalid ROI after clipping: " + ", ".join(invalid_roi_names),
        )

    active = _parse_roi(clipped_roi_cfg, "active_frame_roi")
    ax, ay, aw, ah = active
    safe_border_margin_px = int(config.get("calibration", {}).get("roi_suggestion", {}).get("safe_border_margin_px", 24))
    border_close_roi_names: List[str] = []
    invalid_set = set(invalid_roi_names)
    for k in roi_items:
        if k in invalid_set:
            continue
        r = roi_suggestion["result"].get(k, {})
        rx, ry, rw, rh = int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("w", 0)), int(r.get("h", 0))
        left_d = rx - ax
        top_d = ry - ay
        right_d = (ax + aw) - (rx + rw)
        bottom_d = (ay + ah) - (ry + rh)
        if min(left_d, top_d, right_d, bottom_d) <= safe_border_margin_px:
            border_close_roi_names.append(k)
    # invalid has higher priority than too-close-border for same ROI
    if border_close_roi_names:
        _append_risk(
            "roi_geometry_too_close_border",
            "ROI_GEOMETRY_TOO_CLOSE_BORDER",
            f"ROI too close to active_frame border (margin<={safe_border_margin_px}px): " + ", ".join(border_close_roi_names),
        )

    if roi_suggestion["scores"]["overall_confidence_score"] < 0.65:
        _append_risk(
            "roi_low_score",
            "ROI_LOW_CONFIDENCE_SCORE",
            f"roi_suggestion overall_confidence_score low: {roi_suggestion['scores']['overall_confidence_score']}",
        )

    # direction-vs-ROI geometry conflict check
    dval = direction_info.get("final", {}).get("value")
    er = roi_suggestion["result"].get("entry_roi", {})
    xr = roi_suggestion["result"].get("exit_roi", {})
    e_cx = int(er.get("x", 0)) + int(er.get("w", 0)) / 2.0
    x_cx = int(xr.get("x", 0)) + int(xr.get("w", 0)) / 2.0
    geometry_direction_conflict = (dval == "left_to_right" and e_cx > x_cx) or (dval == "right_to_left" and e_cx < x_cx)

    # window-level background-bias risk (global context, not just final candidate_count)
    selected_window_ids = {int(x) for x in ex.get("selected_event_window_ids", []) if isinstance(x, (int, float))}
    event_windows = ex.get("event_windows", []) if isinstance(ex.get("event_windows", []), list) else []
    selected_windows = [w for w in event_windows if int(w.get("window_id", -1)) in selected_window_ids] if selected_window_ids else []
    if not selected_windows and event_windows:
        selected_windows = sorted(event_windows, key=lambda w: float(w.get("event_likeness_score", 0.0)), reverse=True)[:2]

    high_conf = float(ex.get("final_confidence", 0.0)) >= 0.78
    background_context_hits = 0
    strong_background_hit = False
    for w in selected_windows:
        prefilter_context_tracks = int(w.get("prefilter_context_tracks", w.get("candidate_count", 0)))
        quality_filtered_tracks = int(w.get("quality_filtered_tracks", 0))
        crowd_suppressed_tracks = int(w.get("crowd_suppressed_tracks", 0))
        small_ratio = float(w.get("small_target_ratio", 0.0))
        top1_size_rank_proxy = float(w.get("top1_size_rank_proxy", 1.0))
        same_direction_group_count = int(w.get("same_direction_group_count", 0))

        weak_subjectness = top1_size_rank_proxy < 0.72
        context_dense = prefilter_context_tracks >= 4
        suppressed_heavy = (quality_filtered_tracks + crowd_suppressed_tracks) >= 3
        flow_grouped = same_direction_group_count >= 2

        hit = (context_dense and (suppressed_heavy or small_ratio >= 0.55)) or (flow_grouped and weak_subjectness and small_ratio >= 0.45)
        if hit:
            background_context_hits += 1
        if prefilter_context_tracks >= 6 and (crowd_suppressed_tracks >= 1 or quality_filtered_tracks >= 3):
            strong_background_hit = True

    selected_count = len(selected_windows)
    background_hit_ratio = background_context_hits / max(1, selected_count)
    background_biased_windows = direction_reliable and selected_count > 0 and (
        strong_background_hit or (high_conf and background_hit_ratio >= 0.5)
    )
    if background_biased_windows:
        _append_risk(
            "direction_background_biased_event_windows",
            "DIRECTION_BACKGROUND_BIASED_EVENT_WINDOWS",
            (
                f"selected windows show dense/suppressed background context: hit_ratio={round(background_hit_ratio,3)}, "
                f"strong_hit={strong_background_hit}, selected_windows={selected_count}"
            ),
        )

    # track-level weak-foregroundness risk (distinct from window-level conditions)
    topk_items = ex.get("top_k_selected", []) if isinstance(ex.get("top_k_selected", []), list) else []
    weak_track_votes = 0
    for item in topk_items:
        feats = item.get("features", {}) if isinstance(item, dict) else {}
        med_area = float(feats.get("median_area", 0.0))
        med_h = float(feats.get("median_height", 0.0))
        sparse = float(feats.get("sparse_score", 0.0))
        core_dist = float(feats.get("core_distance_ratio", 0.0))
        weak_conditions = [
            med_area < 2200.0,
            med_h < 110.0,
            sparse > 0.7,
            core_dist > 0.55,
        ]
        if sum(1 for x in weak_conditions if x) >= 2:
            weak_track_votes += 1
    weak_topk_ratio = weak_track_votes / max(1, len(topk_items))
    weak_foreground_topk = direction_reliable and len(topk_items) > 0 and weak_topk_ratio >= 0.6
    if weak_foreground_topk:
        _append_risk(
            "direction_topk_weak_foregroundness",
            "DIRECTION_TOPK_WEAK_FOREGROUNDNESS",
            f"top-k tracks show weak foregroundness: weak_ratio={round(weak_topk_ratio,3)}, weak_track_votes={weak_track_votes}, topk={len(topk_items)}",
        )

    weak_profile = float(ex.get("main_subject_profile_confidence", 0.0)) < 0.55
    low_topk = int(ex.get("voting_tracks", 0)) < max(2, int(ex.get("top_k_limit", 0) // 2) if int(ex.get("top_k_limit", 0)) > 0 else 2)
    auto_status = str(direction_info.get("auto_inference", {}).get("status", ""))
    manual_review_min_voting_tracks = int(
        ex.get("min_voting_tracks")
        or config.get("calibration", {}).get("direction_auto", {}).get("min_voting_tracks", 2)
    )
    insufficient_evidence = (not direction_reliable) and (
        auto_status == "failed_insufficient_candidates" or int(ex.get("voting_tracks", 0)) < manual_review_min_voting_tracks
    )

    conflict_signal = (weak_profile or low_topk or geometry_direction_conflict or background_biased_windows)
    if high_conf and conflict_signal and direction_reliable and not insufficient_evidence:
        _append_risk(
            "direction_confident_but_event_profile_conflict",
            "DIRECTION_CONFIDENT_BUT_EVENT_PROFILE_CONFLICT",
            (
                f"direction confidence appears high ({ex.get('final_confidence', 0)}), and evidence is sufficient; "
                f"event profile/geometry conflicts (profile={ex.get('main_subject_profile_confidence', 0)}, "
                f"voting_tracks={ex.get('voting_tracks', 0)}, geometry_conflict={geometry_direction_conflict}, "
                f"background_biased_windows={background_biased_windows})."
            ),
        )
    elif high_conf and conflict_signal and insufficient_evidence:
        manual_reasons.append({
            "code": "DIRECTION_EVIDENCE_INSUFFICIENT_OR_UNRELIABLE",
            "message": (
                f"direction confidence value may be pseudo-high under insufficient evidence; "
                f"status={auto_status}, voting_tracks={ex.get('voting_tracks', 0)}, min_voting_tracks={manual_review_min_voting_tracks}."
            ),
        })

    calibration_run_id = f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    priority_decision_trace = _build_priority_decision_trace(
        event_id=f"{calibration_run_id}:manual_review",
        input_window={"start_ts": started_at, "end_ts": _utc_now_iso()},
        risk_flags=risk_flags,
    )

    calibration_report = {
        "schema_version": "1.0.0",
        "run_id": calibration_run_id,
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
            "priority": priority_decision_trace["final_priority"],
            "priority_decision_trace": priority_decision_trace,
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
    _export_ml_ready_artifacts(config, calibration_report, direction_info, input_video_path)

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
