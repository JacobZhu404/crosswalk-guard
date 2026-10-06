# 斑马线行人绿灯违章占道检测系统

自动检测「**斑马线行人绿灯（或闪烁清空相位）时，车辆仍静止压在斑马线上阻碍行人过街**」的违章行为，识别车牌，输出标注视频 + 违章事件表 + 证据截图。

> **刚接手这个项目？先读 [`HANDOFF.md`](HANDOFF.md)（交接总纲）。**
> 用 AI 助手继续开发？让它先读 [`docs/AGENT_GUIDE.md`](docs/AGENT_GUIDE.md)（红线与已验证的死路，防止重踩坑）。

---

## 这个系统的定位

- **纯 CV + 轻量模型**，不依赖大模型（VLM）：车辆 YOLOv8n，红绿灯 HSV 颜色法，车牌 HyperLPR3。
- **CPU 可跑**，面向本地 Windows 批量处理手机录制视频。结果确定性、可复现、可解释。
- **违章语义**（唯一权威定义，E12 反转后）：
  **违章 = 行人绿灯/闪烁　∧　车辆静止　∧　车辆压斑马线　且持续足够时长。**
  🔴 红灯不算违章；❓ 没拍到灯默认不判（斑马线疑似遮挡时降为「待复核」）。
  权威规格：[`docs/plans/2026-07-12-design-requirements-v2.md`](docs/plans/2026-07-12-design-requirements-v2.md)。

## 当前成绩（全 11 视频真跑复现）

| 口径 | 指标 |
|---|---|
| 窗级（每次过街窗） | **F1=0.941　P=1.000　R=0.889** |
| 车级（每辆违章车） | 具名召回=0.583　**误罚=0** |
| 车牌 | 命中 5/7 |

**判断违章已经很稳、精度 100%、零误罚**；主要提升空间在模糊车牌识别。详见 `HANDOFF.md`。

## 快速开始（Windows）

```bash
# 1. 装环境(一次)
#    - Python 3.9~3.12
#    - Visual C++ 2015-2022 运行库(torch 加载必需)
pip install -r requirements.txt
pip install ultralytics hyperlpr3 python-docx

# 2. 放视频
#    把手机视频放进 input_video/

# 3. 处理单个视频 → 带标记结果视频 + 违章表
python scripts/run_video.py input_video/某视频.mp4 data/output/结果目录

# 4. 批量评测(确认整体成绩)
python scripts/eval_violations.py --detector v2 --occ-denom box
#    应得 F1=0.941 / P=1.000 / 车级误罚=0

# 5. 生成逐视频报告 + 带标注结果视频(交付物)
python scripts/render_report.py
```

> 大文件（模型权重 `models/` 等）不在 git 里，通过网盘压缩包交付，解压覆盖到项目根目录即可。详见 `HANDOFF.md` 第一节。

## 输出说明

`--output` 目录下：
- `annotated.mp4`：带标注视频（斑马线青色填充、车辆框+ID、静止标黄、违章标红、灯态 HUD、车牌框）。
- `violations.csv`：事件表（含 `status` confirmed/review、时间段、车牌、多牌 `plates`、证据图路径）。
  - `confirmed` = 行人绿灯/闪烁 + 静止 + 压线，三条件同时成立且持续足够时长。
  - `review` = 压线+静止但灯态未知且斑马线疑似遮挡，交人工复核。
- `evidence/`：每起事件的证据截图。

## 灵敏度预设

`--preset strict|balanced|loose|very_loose`（默认 `balanced`）。阈值定义在 `src/redlight/pipeline/tracker.py` 的 `SENSITIVITY_PRESETS`，**不在 config.yaml 里**。

## 目录地图

```
HANDOFF.md               交接总纲(先读这个)
docs/AGENT_GUIDE.md      AI 助手必读: 红线 + 已验证的死路
docs/plans/              2份权威规格(需求 + GT格式)
docs/manuals/            Word 说明书(专业版/科普版/开发过程)
docs/history/            176份过程文档归档(追溯"为什么这么做")
src/redlight/            生产代码(推理流水线本体)
  pipeline/              DAG/跟踪/违章状态机/时序融合/判定
  models/                车辆/斑马线/红绿灯/车牌检测器
  app/cli.py             主入口 + 车牌回填三重约束(0误罚闸门)
  evaluation/            评测框架 + 画廊生成器
scripts/                 66个保留脚本(评测/GT构建/训练/标定/文档生成)
scripts/legacy/          46个一次性诊断脚本(历史问题定位记录)
configs/config.yaml      全部可调参数(检测器版本/阈值/先验)
datasets/gt/             真值: events.csv(事件级) + light_canonical_gt.json(逐帧灯态)
tests/                   单元+集成测试(pytest)
```

## 开发纪律（重要）

- **「0 误罚」是铁底线**：任何改动保持全 11 视频误罚=0。改车牌逻辑前先跑 `python -m pytest tests/`。
- **改动必须全 11 视频真跑验证**，指标不回退。别凭感觉调参。
- **先诊断后改码**：定位问题用只读诊断脚本（`scripts/legacy/` 有大量先例），量化根因再动生产码。
- 完整红线清单见 `docs/AGENT_GUIDE.md`。

---
*本项目曾由多个 AI agent 协同开发，现收敛为单一 main 分支交接。完整开发历史保留在 git 提交记录与 `docs/history/`。*
