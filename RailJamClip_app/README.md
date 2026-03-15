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
4. 输出检测调试 JSON（每帧 person 检测框）
5. 输出符合 required 模板的 `metadata.json`（当前 `events` 为空）

尚未实现（后续）：
- tracking
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
- `output/detections_debug.json`：检测链路调试输出（若 `debug.export_detection_json=true`）
- `output/clips/` 与 `logs/` 目录会被创建

## 配置重点
- `detector.frame_step`：检测步长（`1`=逐帧，`2`=每2帧一次）
- `debug.export_detection_json`：是否导出调试 JSON
- `debug.detection_json_path`：调试 JSON 输出路径

- `detector.imgsz`：YOLO 推理尺寸，默认 `640`（CPU 推荐起步值）
- 运行时会打印最小进度日志（每处理 50 帧打印一次 `当前帧/总帧数`）
