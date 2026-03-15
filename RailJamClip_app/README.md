# RailJamClip_app

## 项目命名与仓库设定
- 当前仓库与项目显示名统一为：`RailJamClip_app`
- 配置字段统一使用：`project.project_name`
- 运行输出中的 metadata `run.project_name` 也统一写为 `RailJamClip_app`（若配置缺失则自动回退到当前项目目录名）

## 项目目标
固定机位滑雪视频自动切片：从本地 mp4 中检测人物，基于 `entry/core/exit` 三段 ROI 与方向一致性过滤，自动导出事件 clips，并生成 `metadata.json`。

## 当前 MVP 范围
本仓库第一版仅覆盖以下能力：

1. 读取本地 MP4 视频
2. 使用 Ultralytics YOLO 检测 `person`（后续实现）
3. 使用最小 tracking（IOU 优先，中心点距离兜底，后续实现）
4. 使用三段 ROI 状态机判定事件（后续实现）
   - `entry_roi / core_roi / exit_roi` 永远表示事件顺序语义
   - `direction` 仅用于 x 方向一致性过滤
5. 自动导出事件 clips（后续实现）
6. 生成 `metadata.json`

不在第一版范围：UI、人物分组高级策略、颜色过滤、实时摄像头接入。

## 运行前准备
1. 准备 Python 3.10+
2. 安装 `requirements.txt` 中依赖（本轮不执行安装）
3. 准备输入视频，例如：`input/demo.mp4`
4. 复制配置：
   - `cp config.example.yaml config.yaml`
   - 按实际路径调整 `input.video_path`

## 本轮可运行能力（无事件占位运行）
当前 `main.py` 已可完成一轮基础流程：

- 读取 `config.yaml`
- 校验输入视频路径是否存在
- 读取视频基础信息（fps、total_frames、width、height、duration_seconds）
- 创建 `output/clips` 与 `logs` 目录
- 输出符合 required 模板的 `output/metadata.json`
- 当前未实现检测与事件逻辑，因此 `events` 为空数组

运行命令：

```bash
python main.py --config config.yaml
```

## 预期输出
运行后在输出目录得到：

- `output/clips/`：目录存在（本轮可能为空）
- `output/metadata.json`：本次运行的结构化结果

`metadata.json` 将包含 required 顶层字段：

- `schema_version`
- `run`
- `input_video`
- `roi`
- `tracking`
- `event_params`
- `summary`（含 `candidate_events` / `qualified_events` / `exported_clips` / `dropped_events`）
- `events`
