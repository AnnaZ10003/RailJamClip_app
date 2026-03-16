# RailJamClip_app

## 项目命名与仓库设定
- 当前仓库与项目显示名统一为：`RailJamClip_app`
- 配置字段统一使用：`project.project_name`
- 运行输出中的 metadata `run.project_name` 也统一写为 `RailJamClip_app`（若配置缺失则自动回退到当前项目目录名）

## 项目目标
固定机位滑雪视频自动切片：从本地 mp4 中检测人物，基于 `entry/core/exit` 三段 ROI 与方向一致性过滤，自动导出事件 clips，并生成 `metadata.json`。

## 当前 MVP 范围
本仓库第一版当前已覆盖：

1. 读取本地 MP4 视频
2. 使用 Ultralytics YOLO 对视频逐帧或按步长执行检测
3. 仅保留 `person` 类检测框
4. 基于 IOU 优先 + 中心点距离兜底 + 短时连续性实现最小 tracking
5. 使用 `active_frame_roi` 排除两侧黑边干扰
6. 使用 `tracking_roi`（由 entry/core/exit 外包矩形+margin 自动生成）过滤背景人物
7. 输出检测+tracking 调试 JSON（每帧 person 框与 track_id）
8. 输出 tracking 调试预览视频（bbox/track_id/confirmed + ROI）
9. 输出首帧 ROI 标定图 `output/calibration_preview.jpg`
10. 输出符合 required 模板的 `metadata.json`（当前 `events` 为空）

尚未实现（后续）：
- entry/core/exit 状态机
- clips 导出
- UI、颜色过滤、实时流

## 运行前准备
1. 准备 Python 3.10+
2. 安装 `requirements.txt` 中依赖
3. 准备输入视频，例如：`input/demo.mp4`
4. 复制配置：
   - `cp config.example.yaml config.yaml`
   - 按实际路径调整 `input.video_path`

## 运行
```bash
python main.py --config config.yaml
```

## 本轮输出
> 注：当前版本重点放在自动标定可靠性（active_frame 与 direction 降级链路）修复。

- `output/calibration_preview.jpg`：首帧标定图（整帧边界 + active/entry/core/exit + 坐标文字）
- `output/metadata.json`：required 顶层结构（summary 四个必填计数存在）
- `output/detections_debug.json`：检测+tracking 调试 JSON（若 `debug.export_detection_json=true`）
- `output/preview_tracking.mp4`：tracking 可视化预览（若 `debug.export_preview_video=true`）
- `output/clips/` 与 `logs/` 目录会被创建

## 配置重点
- `roi.active_frame_roi`：有效画面区域（先裁剪到视频边界；预览画的是裁剪后结果）
- `roi.entry_roi/core_roi/exit_roi`：第一版使用整帧坐标配置，再裁剪到 `active_frame_roi` 内
- `calibration.export_calibration_preview`：是否导出首帧标定图
- `calibration.preview_image_path`：标定图输出路径
- `tracking.tracking_roi_margin_px`：自动生成走廊 ROI 外扩边距（在 active_frame_roi 内截断）
- `tracking.min_motion_frames/min_motion_distance_px/direction_min_progress_px`：仅用于候选轨迹升级 confirmed 前的运动过滤（本版默认先放松）
- `tracking.core_reacquire_max_frames/core_reacquire_max_dist_px`：仅对 confirmed 且 entered_core 的丢失轨迹保活重连
- ROI 会在加载后自动裁剪到边界；若被裁剪或过小会输出 warning

## Priority 决策 Trace（MVP 10 字段）定义规范

> 目标：第一版仅覆盖最小可用的可解释性闭环，字段总量控制在 10 个。

### 字段清单（仅 MVP）

| 字段名 | 类型 | 必填 | 合法值/枚举 | 示例值 | 校验规则 |
|---|---|---|---|---|---|
| `trace_version` | `string` | 是 | 建议固定为 `"1.0"`（后续可迭代 `1.x`） | `"1.0"` | 必须匹配 `^\\d+\\.\\d+$`；第一版必须为 `"1.0"` |
| `event_id` | `string` | 是 | 业务侧事件唯一标识；允许字母/数字/下划线/中划线 | `"evt_2026_03_16_000123"` | 长度 `1~128`；建议匹配 `^[A-Za-z0-9_-]+$`；同一数据域内唯一 |
| `timestamp` | `string`（date-time） | 是 | ISO8601 UTC 时间戳 | `"2026-03-16T09:15:30Z"` | 必须为 RFC3339/ISO8601 格式；建议统一 UTC（`Z` 结尾） |
| `input_window` | `object` | 是 | `start_ts`、`end_ts` 两个子字段（均为 date-time） | `{ "start_ts": "2026-03-16T09:14:30Z", "end_ts": "2026-03-16T09:15:30Z" }` | `start_ts <= end_ts`；时间窗不得为空 |
| `source` | `object` | 是 | 至少包含 `engine`、`engine_version` | `{ "engine": "priority-rule-engine", "engine_version": "0.1.0" }` | `engine` 非空字符串；`engine_version` 建议 SemVer（如 `0.1.0`） |
| `severity_counts` | `object` | 是 | 第一版固定键：`critical/high/medium/low/info/total`（均为整数） | `{ "critical":0, "high":3, "medium":6, "low":2, "info":0, "total":11 }` | 各值为 `>=0` 整数；`total = critical + high + medium + low + info` |
| `triggered_rules` | `array<object>` | 是 | 每项建议包含：`rule_id`、`matched`、`proposed_priority`、`explanation` | `[{ "rule_id":"R_HIGH_COUNT_THRESHOLD", "matched":true, "proposed_priority":"high", "explanation":"high >= 3" }]` | 第一版建议仅记录命中规则；建议限制条数 `<= 5`（防止 payload 过重）；至少 1 条 |
| `primary_rule` | `object` | 是 | 建议包含：`rule_id`、`decision_basis`、`decisive_evidence` | `{ "rule_id":"R_HIGH_COUNT_THRESHOLD", "decision_basis":"Matched highest-rank rule", "decisive_evidence": { "severity_counts.high":3, "threshold":3 } }` | `rule_id` 必须出现在 `triggered_rules.rule_id` 中；`decision_basis` 非空 |
| `final_priority` | `string` | 是 | 枚举：`high` / `medium` / `low` | `"high"` | 严格枚举校验；禁止空值与大小写变体 |
| `decision_path` | `string` | 是 | 人类可读单行路径 | `"R_HIGH_COUNT_THRESHOLD -> high"` | 长度建议 `1~256`；应包含主规则 `rule_id` 与 `final_priority` |

### `triggered_rules` 轻量化建议（第一版）

- 建议**只记录命中规则**，不记录全部候选规则，以降低存储与传输成本。
- 建议设置最大条数（推荐 `<=5`）：
  - 第一版可解释性只需看到“触发过哪些关键规则”。
  - 条数过多会导致 trace 噪音增大，不利于业务方快速阅读。
- 若命中规则超过上限，可仅保留前 N 条（按规则优先级或执行顺序），并在后续审计版再补全。

### 为什么这 10 个字段足够第一版可解释性

- `event_id/timestamp/input_window/source`：回答“是谁、何时、基于哪段输入、由谁判定”。
- `severity_counts/triggered_rules/primary_rule`：回答“依据是什么，哪条规则拍板”。
- `final_priority/decision_path`：回答“最终结论是什么、路径是什么”。

以上正好覆盖“可追溯 + 可复核 + 可说明”的最小闭环，不引入审计级复杂度。
