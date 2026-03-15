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
7. 使用最小 bbox 尺寸过滤减少小目标噪声
8. 输出检测+tracking 调试 JSON（每帧 person 框与 track_id）
9. 输出 tracking 调试预览视频（bbox/track_id/confirmed + ROI）
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
- `output/metadata.json`：required 顶层结构（summary 四个必填计数存在）
- `output/detections_debug.json`：检测+tracking 调试 JSON（若 `debug.export_detection_json=true`）
- `output/preview_tracking.mp4`：tracking 可视化预览（若 `debug.export_preview_video=true`）
- `output/clips/` 与 `logs/` 目录会被创建

## 配置重点
- `roi.active_frame_roi`：有效画面区域（先裁剪到视频边界；预览画的是裁剪后结果）
- `roi.entry_roi/core_roi/exit_roi`：第一版使用整帧坐标配置，再裁剪到 `active_frame_roi` 内
- `tracking.tracking_roi_margin_px`：自动生成走廊 ROI 外扩边距（在 active_frame_roi 内截断）
- `tracking.min_bbox_width_px/min_bbox_height_px/min_bbox_area_px`：进入 tracking 的最小框过滤
- `tracking.min_motion_frames/min_motion_distance_px/direction_min_progress_px`：仅用于候选轨迹升级 confirmed 前的运动过滤
- `tracking.core_reacquire_max_frames/core_reacquire_max_dist_px`：仅对 confirmed 且 entered_core 的丢失轨迹保活重连
- `detector.frame_step`：检测/追踪步长（默认 `2`，更适合 CPU 测试）
- `detector.imgsz`：YOLO 推理尺寸，默认 `640`
- ROI 会在加载后自动裁剪到边界；若被裁剪或过小会输出 warning
