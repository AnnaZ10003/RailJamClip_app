"""Main entry for RailJamClip_app MVP.

当前实现范围（本轮）：
- 读取 config.yaml
- 校验输入视频路径
- 读取视频基础信息
- 创建 output/clips 与 logs 目录
- 生成并写出“无事件占位运行”的 metadata.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from metadata import build_metadata, write_metadata
from utils import ensure_dir, load_config, read_video_info


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
    """根据当前文件所在目录推断仓库/项目名。

    说明：
    - 当前代码位于 `<repo_root>/RailJamClip_app/main.py`
    - 若配置未提供 `project.project_name`，优先使用父目录名作为兜底。
    """
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


def run_pipeline(config_path: Path) -> int:
    """运行本轮最小可运行流程（无事件占位）。"""
    started_at = _utc_now_iso()
    config = load_config(config_path)

    input_video_path = Path(config["input"]["video_path"])
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    video_info = read_video_info(input_video_path)

    clips_dir = Path(config["output"]["clips_dir"])
    debug_log_path = Path(config["output"]["debug_log_path"])
    metadata_path = Path(config["output"]["metadata_path"])

    ensure_dir(clips_dir)
    ensure_dir(debug_log_path.parent)

    # 本轮不做检测/跟踪/事件识别，按无事件占位输出。
    events = []
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
