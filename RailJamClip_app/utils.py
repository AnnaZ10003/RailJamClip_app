"""Utility helpers for RailJamClip_app.

职责：
- 配置读取与基础校验
- 视频基础信息读取
- 时间/帧换算
- 路径创建
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置。

    Args:
        config_path: 配置文件路径。

    Returns:
        dict 配置对象。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config content must be a mapping object.")
    return config


def read_video_info(video_path: Path) -> Dict[str, Any]:
    """读取视频基础信息。

    Args:
        video_path: 输入视频路径。

    Returns:
        包含 fps、total_frames、width、height、duration_seconds 的字典。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration_seconds = frame_to_time(total_frames, fps) if fps > 0 else 0.0
    return {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def frame_to_time(frame_idx: int, fps: float) -> float:
    """帧号转秒。"""
    if fps <= 0:
        return 0.0
    return frame_idx / fps


def time_to_frame(time_sec: float, fps: float) -> int:
    """秒转帧号。"""
    if fps <= 0:
        return 0
    return int(time_sec * fps)


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
