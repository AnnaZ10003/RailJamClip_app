#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SEMANTIC_RULESET_VERSION = "1.0"
DEFAULT_MIN_VOTING_TRACKS = 2


class ValidationIssue:
    def __init__(self, level: str, code: str, location: str, message: str) -> None:
        self.level = level
        self.code = code
        self.location = location
        self.message = message

    def __str__(self) -> str:
        return f"{self.level} {self.code} {self.location} {self.message}"


class ValidationContext:
    def __init__(self, strict: bool = False, max_errors: int = 200) -> None:
        self.strict = strict
        self.max_errors = max_errors
        self.issues: List[ValidationIssue] = []
        self.stats: Dict[str, Any] = {
            "rows": {},
            "direction_risk_semantic_class": Counter(),
            "label_train_eligibility": Counter(),
            "auto_inference_status": Counter(),
            "review_status": {
                "track_samples": Counter(),
                "window_samples": Counter(),
                "video_samples": Counter(),
            },
            "manual_review_recommended_true": 0,
            "manual_reason_empty_when_recommended": 0,
            "warn_drift_candidates": [],
        }

    def add_issue(self, level: str, code: str, location: str, message: str) -> None:
        if code == "SEM101" and self.strict and level == "WARN":
            level = "ERROR"
        self.issues.append(ValidationIssue(level, code, location, message))

    def has_error(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ml_ready semantic constraints.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    parser.add_argument("--track", type=Path, help="Path to track_samples.jsonl")
    parser.add_argument("--window", type=Path, help="Path to window_samples.jsonl")
    parser.add_argument("--video", type=Path, help="Path to video_samples.jsonl")
    parser.add_argument("--strict", action="store_true", help="Upgrade selected warnings to errors")
    parser.add_argument("--max-errors", type=int, default=200, help="Maximum issues to print")
    parser.add_argument("--print-summary", action="store_true", default=True, help="Print summary stats")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid json at {path}: {exc}") from exc



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    raise SystemExit(f"empty line not allowed: {path}:{idx}")
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid jsonl at {path}:{idx}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise SystemExit(f"jsonl row must be object: {path}:{idx}")
                obj["__line__"] = idx
                obj["__path__"] = str(path)
                rows.append(obj)
    except FileNotFoundError as exc:
        raise SystemExit(f"jsonl not found: {path}") from exc
    return rows



def resolve_path(kind: str, cli_value: Optional[Path], manifest: Dict[str, Any], manifest_path: Path) -> Path:
    if cli_value is not None:
        return cli_value
    rel = manifest.get("files", {}).get(kind, {}).get("path")
    if not rel:
        raise SystemExit(f"manifest missing files.{kind}.path")
    p = Path(rel)
    return p if p.is_absolute() else manifest_path.parent / p



def row_location(row: Dict[str, Any]) -> str:
    return f"{row.get('__path__','?')}:{row.get('__line__','?')}"



def is_non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0



def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")



def get_min_voting_tracks(row: Dict[str, Any], manifest: Dict[str, Any]) -> int:
    # row-local preferred sources
    gate_thresholds = row.get("gate_funnel", {}).get("thresholds_applied", {})
    candidates = [
        row.get("min_voting_tracks"),
        row.get("gate_funnel", {}).get("min_voting_tracks"),
        gate_thresholds.get("min_voting_tracks"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and int(candidate) > 0:
            return int(candidate)

    # manifest/config fallback if present
    manifest_candidates = [
        manifest.get("config", {}).get("direction", {}).get("min_voting_tracks"),
        manifest.get("defaults", {}).get("min_voting_tracks"),
    ]
    for candidate in manifest_candidates:
        if isinstance(candidate, (int, float)) and int(candidate) > 0:
            return int(candidate)

    return DEFAULT_MIN_VOTING_TRACKS



def assert_unique(rows: Sequence[Dict[str, Any]], keys: Sequence[str], code: str, ctx: ValidationContext) -> None:
    seen: Dict[Tuple[Any, ...], str] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        loc = row_location(row)
        if key in seen:
            ctx.add_issue("ERROR", code, loc, f"duplicate primary key {keys}={key}; first seen at {seen[key]}")
        else:
            seen[key] = loc



def check_manifest_cross_file_consistency(manifest: Dict[str, Any], track_rows: List[Dict[str, Any]], window_rows: List[Dict[str, Any]], video_rows: List[Dict[str, Any]], ctx: ValidationContext) -> None:
    expected_counts = {
        "track_samples": len(track_rows),
        "window_samples": len(window_rows),
        "video_samples": len(video_rows),
    }
    code_map = {
        "track_samples": "SEM009",
        "window_samples": "SEM010",
        "video_samples": "SEM011",
    }
    for name, actual in expected_counts.items():
        expected = manifest.get("files", {}).get(name, {}).get("rows")
        if expected != actual:
            ctx.add_issue("ERROR", code_map[name], "manifest", f"{name} row count mismatch: manifest={expected}, actual={actual}")

    manifest_run = manifest.get("run_id")
    for rows in (track_rows, window_rows, video_rows):
        for row in rows:
            if row.get("run_id") != manifest_run:
                ctx.add_issue("ERROR", "SEM012", row_location(row), f"run_id mismatch with manifest: row={row.get('run_id')} manifest={manifest_run}")

    assert_unique(track_rows, ["run_id", "video_id", "window_id", "track_id"], "SEM013", ctx)
    assert_unique(window_rows, ["run_id", "video_id", "window_id"], "SEM014", ctx)
    assert_unique(video_rows, ["run_id", "video_id"], "SEM015", ctx)



def check_video_semantics(video_rows: Sequence[Dict[str, Any]], manifest: Dict[str, Any], ctx: ValidationContext) -> None:
    for row in video_rows:
        loc = row_location(row)
        risk_class = row.get("direction_risk_semantic_class")
        eligibility = row.get("label_train_eligibility")
        status = row.get("auto_inference_status")
        reliable = row.get("direction_reliable")
        voting_tracks = int(row.get("voting_tracks", 0) or 0)
        min_voting_tracks = get_min_voting_tracks(row, manifest)

        ctx.stats["direction_risk_semantic_class"][str(risk_class)] += 1
        ctx.stats["label_train_eligibility"][str(eligibility)] += 1
        ctx.stats["auto_inference_status"][str(status)] += 1
        if row.get("manual_review_recommended") is True:
            ctx.stats["manual_review_recommended_true"] += 1

        if risk_class == "evidence_insufficient":
            cond1 = reliable is False
            cond2 = status == "failed_insufficient_candidates"
            cond3 = voting_tracks < min_voting_tracks
            if not (cond1 or cond2 or cond3):
                ctx.add_issue("ERROR", "SEM001", loc, "evidence_insufficient boundary violated")

        if risk_class == "confident_conflict":
            if reliable is not True:
                ctx.add_issue("ERROR", "SEM002", loc, "confident_conflict requires direction_reliable=true")
            if status != "ok":
                ctx.add_issue("ERROR", "SEM003", loc, "confident_conflict requires auto_inference_status=ok")
            if eligibility != "eligible":
                ctx.add_issue("ERROR", "SEM004", loc, "confident_conflict should be eligible for training")

        if eligibility == "weak_only":
            if risk_class != "evidence_insufficient":
                ctx.add_issue("ERROR", "SEM005", loc, "weak_only must map to evidence_insufficient in v1 policy")
            if row.get("final_confidence") is not None:
                ctx.add_issue("WARN", "SEM101", loc, "weak_only row has final_confidence; do not use as training weight")



def check_manual_review_reason_consistency(video_rows: Sequence[Dict[str, Any]], ctx: ValidationContext) -> None:
    for row in video_rows:
        loc = row_location(row)
        recommended = row.get("manual_review_recommended")
        reasons = row.get("manual_reason_codes")
        if recommended is True:
            if not is_non_empty_list(reasons):
                ctx.stats["manual_reason_empty_when_recommended"] += 1
                ctx.add_issue("ERROR", "SEM006", loc, "manual_review_recommended=true but manual_reason_codes is empty")
        else:
            if is_non_empty_list(reasons):
                ctx.add_issue("WARN", "SEM102", loc, "manual_review_recommended=false but reasons are present")



def check_review_consistency(rows: Sequence[Dict[str, Any]], dataset_name: str, ctx: ValidationContext) -> None:
    for row in rows:
        loc = row_location(row)
        status = row.get("review_status")
        reviewer_id = row.get("reviewer_id")
        reviewed_at = row.get("reviewed_at")
        ctx.stats["review_status"][dataset_name][str(status)] += 1
        if status == "unreviewed":
            if reviewer_id is not None or reviewed_at is not None:
                ctx.add_issue("ERROR", "SEM007", loc, "unreviewed rows must have reviewer_id/reviewed_at = null")
        else:
            if is_empty(reviewer_id) or reviewed_at is None:
                ctx.add_issue("ERROR", "SEM008", loc, "reviewed rows must have reviewer_id and reviewed_at")



def check_drift_warnings(video_rows: Sequence[Dict[str, Any]], ctx: ValidationContext) -> None:
    if not video_rows:
        return
    failed_insufficient = sum(1 for r in video_rows if r.get("auto_inference_status") == "failed_insufficient_candidates")
    weak_only = sum(1 for r in video_rows if r.get("label_train_eligibility") == "weak_only")
    median_like = sorted(int(r.get("voting_tracks", 0) or 0) for r in video_rows)
    median_voting = median_like[len(median_like) // 2]

    ratio_failed = failed_insufficient / len(video_rows)
    ratio_weak = weak_only / len(video_rows)
    if ratio_failed > 0.5:
        ctx.add_issue("WARN", "SEM201", "video_samples", f"failed_insufficient_candidates ratio is high: {ratio_failed:.2%}")
    if ratio_weak > 0.5:
        ctx.add_issue("WARN", "SEM202", "video_samples", f"weak_only ratio is high: {ratio_weak:.2%}")
    if median_voting < 2:
        ctx.add_issue("WARN", "SEM203", "video_samples", f"median voting_tracks is low: {median_voting}")



def print_issues(issues: Sequence[ValidationIssue], max_errors: int) -> None:
    for issue in issues[:max_errors]:
        print(str(issue))
    if len(issues) > max_errors:
        print(f"... truncated {len(issues) - max_errors} additional issues")



def print_summary(ctx: ValidationContext) -> None:
    print("=== validate_semantics summary ===")
    print(f"semantic_ruleset_version: {SEMANTIC_RULESET_VERSION}")
    print(f"errors: {sum(1 for i in ctx.issues if i.level == 'ERROR')}")
    print(f"warnings: {sum(1 for i in ctx.issues if i.level == 'WARN')}")
    print("direction_risk_semantic_class:", dict(ctx.stats["direction_risk_semantic_class"]))
    print("label_train_eligibility:", dict(ctx.stats["label_train_eligibility"]))
    print("auto_inference_status:", dict(ctx.stats["auto_inference_status"]))
    print("review_status:", {k: dict(v) for k, v in ctx.stats["review_status"].items()})
    print("manual_review_recommended_true:", ctx.stats["manual_review_recommended_true"])
    print("manual_reason_empty_when_recommended:", ctx.stats["manual_reason_empty_when_recommended"])



def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    ctx = ValidationContext(strict=args.strict, max_errors=args.max_errors)

    track_rows = load_jsonl(resolve_path("track_samples", args.track, manifest, args.manifest))
    window_rows = load_jsonl(resolve_path("window_samples", args.window, manifest, args.manifest))
    video_rows = load_jsonl(resolve_path("video_samples", args.video, manifest, args.manifest))

    ctx.stats["rows"] = {
        "track_rows": len(track_rows),
        "window_rows": len(window_rows),
        "video_rows": len(video_rows),
    }

    check_manifest_cross_file_consistency(manifest, track_rows, window_rows, video_rows, ctx)
    check_video_semantics(video_rows, manifest, ctx)
    check_review_consistency(track_rows, "track_samples", ctx)
    check_review_consistency(window_rows, "window_samples", ctx)
    check_review_consistency(video_rows, "video_samples", ctx)
    check_manual_review_reason_consistency(video_rows, ctx)
    check_drift_warnings(video_rows, ctx)

    print_issues(ctx.issues, args.max_errors)
    if args.print_summary:
        print_summary(ctx)

    return 1 if ctx.has_error() else 0


if __name__ == "__main__":
    raise SystemExit(main())
