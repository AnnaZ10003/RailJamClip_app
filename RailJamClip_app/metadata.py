"""Metadata generation for RailJamClip_app MVP.

职责：
- 组装并写出 metadata.json
- 保证 required 顶层字段与 required 统计字段存在
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_metadata(
    schema_version: str,
    run_info: Dict[str, Any],
    input_video: Dict[str, Any],
    roi: Dict[str, Any],
    tracking: Dict[str, Any],
    event_params: Dict[str, Any],
    summary: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建 metadata 顶层结构。"""
    return {
        "schema_version": schema_version,
        "run": run_info,
        "input_video": input_video,
        "roi": roi,
        "tracking": tracking,
        "event_params": event_params,
        "summary": summary,
        "events": events,
    }


def validate_required_fields(metadata: Dict[str, Any]) -> None:
    """校验 MVP required 字段是否齐全。"""
    required_top = [
        "schema_version",
        "run",
        "input_video",
        "roi",
        "tracking",
        "event_params",
        "summary",
        "events",
    ]

    missing_top = [k for k in required_top if k not in metadata]
    if missing_top:
        raise ValueError(f"metadata missing required top fields: {missing_top}")

    required_summary = [
        "candidate_events",
        "qualified_events",
        "exported_clips",
        "dropped_events",
    ]
    summary = metadata.get("summary", {})
    missing_summary = [k for k in required_summary if k not in summary]
    if missing_summary:
        raise ValueError(f"metadata.summary missing required fields: {missing_summary}")


def write_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """写出 metadata.json。"""
    validate_required_fields(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
