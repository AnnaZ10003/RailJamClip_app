"""Main entry for RailJamClip_app MVP.

当前实现范围（本轮）：
- 读取 config.yaml
- 校验输入视频路径
- 读取视频基础信息
- 创建 output/clips 与 logs 目录
- 接入 YOLO person 检测并统计检测结果
- 可选输出每帧检测框调试 JSON
- 生成并写出 metadata.json（当前不做 tracking/roi_event/clips）
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2

from detector import Detection, PersonDetector
from metadata import build_metadata, write_metadata
from utils import ensure_dir, load_config, read_video_info, write_json


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RailJamClip_app MVP runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config yaml (default: config.yaml)",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _detect_repo_name_fallback() -> str:
    """根据当前文件所在目录推断仓库/项目名。"""
    return Path(__file__).resolve().parent.name


def _build_run_info(config: Dict[str, Any], started_at: str, finished_at: str, status: str) -> Dict[str, Any]:
    """构建 metadata.run 字段。"""
    project_name = config.get("project", {}).get("project_name") or _detect_repo_name_fallback()
    run_name = config.get("project", {}).get("run_name", "default_run")
    return {
        "project_name": project_name,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
    }


def _to_debug_item(frame_idx: int, detections: List[Detection]) -> Dict[str, Any]:
    """将单帧检测结果转换为调试 JSON 结构。"""
    return {
        "frame_index": frame_idx,
        "person_count": len(detections),
        "detections": [
            {
                "bbox_xyxy": [d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]],
                "confidence": d.confidence,
                "class_id": d.class_id,
            }
            for d in detections
        ],
    }


def run_pipeline(config_path: Path) -> int:
    """运行本轮检测链路 MVP（无事件占位）。"""
    started_at = _utc_now_iso()
    config = load_config(config_path)
    print(f"[INFO] 已读取配置: {config_path}")

    input_video_path = Path(config["input"]["video_path"])
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    video_info = read_video_info(input_video_path)
    print(f"[INFO] 已打开视频: {input_video_path}")
    print(
        "[INFO] 视频基础信息: "
        f"fps={video_info['fps']}, total_frames={video_info['total_frames']}, "
        f"width={video_info['width']}, height={video_info['height']}, "
        f"duration_seconds={video_info['duration_seconds']:.3f}"
    )

    clips_dir = Path(config["output"]["clips_dir"])
    debug_log_path = Path(config["output"]["debug_log_path"])
    metadata_path = Path(config["output"]["metadata_path"])
    debug_json_enabled = bool(config.get("debug", {}).get("export_detection_json", True))
    debug_json_path = Path(config.get("debug", {}).get("detection_json_path", "output/detections_debug.json"))

    ensure_dir(clips_dir)
    ensure_dir(debug_log_path.parent)

    detector_cfg = config.get("detector", {})
    frame_step = int(detector_cfg.get("frame_step", 1))
    if frame_step <= 0:
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

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video for detection: {input_video_path}")

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

        # 安全抽样：仅对抽样帧调用 YOLO，非抽样帧直接跳过。
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        detections = detector.predict_frame(frame)
        processed_frames += 1
        total_person_boxes += len(detections)
        if detections:
            frames_with_person += 1

        if debug_json_enabled:
            debug_frames.append(_to_debug_item(frame_idx, detections))

        if processed_frames % 50 == 0:
            current_frame = frame_idx + 1
            print(f"[INFO] 检测进度: {current_frame}/{total_frames}")

        frame_idx += 1

    cap.release()

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

    # 本轮不做 tracking / roi_event / clips。
    events: List[Dict[str, Any]] = []
    summary = {
        "candidate_events": 0,
        "qualified_events": 0,
        "exported_clips": 0,
        "dropped_events": 0,
    }

    metadata = build_metadata(
        schema_version="1.0.0",
        run_info=_build_run_info(config, started_at=started_at, finished_at=_utc_now_iso(), status="success"),
        input_video=video_info,
        roi=config.get("roi", {}),
        tracking=config.get("tracking", {}),
        event_params=config.get("event", {}),
        summary=summary,
        events=events,
    )
    write_metadata(metadata, metadata_path)
    return 0


def main() -> int:
    """程序入口。"""
    args = parse_args()
    return run_pipeline(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
