"""Clip export skeleton.

职责：
- 根据事件时间区间导出片段
- 支持 ffmpeg / opencv 两种导出方式（MVP 先留接口）
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ExportMethod = Literal["ffmpeg", "opencv"]


def export_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    method: ExportMethod,
    overwrite: bool = True,
    ffmpeg_video_codec: str = "libx264",
    ffmpeg_audio_codec: str = "aac",
) -> None:
    """导出单个 clip。

    TODO:
        - 参数合法性检查
        - dispatch 到 ffmpeg 或 opencv 实现
    """
    raise NotImplementedError


def export_with_ffmpeg(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    overwrite: bool,
    ffmpeg_video_codec: str,
    ffmpeg_audio_codec: str,
) -> None:
    """使用 ffmpeg 命令导出。

    TODO:
        - 构建命令并执行
        - 错误处理
    """
    raise NotImplementedError


def export_with_opencv(video_path: Path, output_path: Path, start_time: float, end_time: float) -> None:
    """使用 OpenCV 逐帧写出片段。

    TODO:
        - 按 fps 计算帧区间
        - VideoWriter 输出
    """
    raise NotImplementedError
