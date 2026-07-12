# 斑马线行人绿灯压线检测工程

纯 CV + 轻量模型方案（**不依赖 VLM 大模型**），面向**本地 Windows + CPU** 批量处理手机录制视频。
目标：检测"**斑马线行人绿灯（或闪烁清空相位）时，车辆仍静止压在斑马线上阻碍行人过街**"的违规行为，并识别车牌号，输出标注视频 + 违规事件报告 + 证据截图（+ 规划中的 COT 小作文）。

> ⚠️ **语义更正（2026-07-11）**：本项目检测的违规 = **行人有路权（绿灯/闪烁）时车辆占道**。  
> 🔴 红灯 = 车辆可通行，**不算违规**；❓ 未拍到灯 = 默认不判违规（仅当斑马线被遮挡时降为待复核）。  
> 旧版"红灯压线=违规"的语义是**错的**，已废弃（详见设计文档 §5.3.1）。

---

## 1. 为什么不用 VLM

- 本场景本质是**几何 + 状态机判定**（行人绿灯 ∧ 车静止 ∧ 车压斑马线），用检测 + 跟踪 + 规则即可稳定、可复现地解决。
- VLM 推理慢、结果有随机性、成本高，不适合做"判定"这种硬规则任务。
- 本方案全部为确定性 CV / 轻量模型推理，CPU 可跑，结果稳定可解释。

---

## 2. 技术架构

```
视频帧
  ├─ 车辆检测     YOLOv8n (ultralytics, COCO car/bus/truck) + IoU 跟踪
  ├─ 斑马线检测   经典 CV 兜底 v11 (多位置条带扫描 + 车辆锚定); 可选分割模型
  ├─ 红绿灯检测   颜色兜底 v5 (亮斑 + 均值色分类) + 时间平滑; 可选模型
  ├─ 车牌识别     HyperLPR3 (HIGH) + 多帧加权投票(按 track_id 全局聚合)
  ├─ 静止判定     TrackStateManagerV2 滑动窗口速度(抗抖动)
  └─ 违规状态机   行人绿灯/闪烁 ∧ 静止 ∧ 压斑马线(footprint IoU) 且持续 duration 帧
        ↓
  输出: annotated.mp4 + violations.csv + evidence/*.jpg
```

**关键设计（针对你的约束）：**
- **CPU-only** → 全用 nano 级模型 + 降采样推理（默认 8fps 足够抓静止车）。
- **机位不固定** → 斑马线/红绿灯逐帧检测，移动拍摄也能 work。
- **部分路段拍不到红绿灯** → 灯态返回 `unknown`，相关疑似事件归入 **待复核 (status=review)**，绝不瞎判。
- **能力解耦（2026-07-11 新增）**：最终结论 = 红绿灯识别 × 静止占道 × 车牌识别，三项能力各自准确才可能综合正确；因此评测与优化按能力**独立拆解**（见 §7）。

---

## 3. 环境安装

需要本机有 Python 3.9~3.12，并安装 [VC++ 2015-2022 运行库](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)（torch/onnxruntime 加载前提）。

```bash
cd redlight-crosswalk-violation
pip install -r requirements.txt
```

可选：Intel CPU 加速（提速明显）
```bash
pip install openvino openvino-dev
# 然后把 yolov8n.pt 导出为 OpenVINO: yolo export model=yolov8n.pt format=openvino
```

---

## 4. 运行

### 4.1 准备输入视频
把手机视频放到 `input_video/`（如 `input_video/违章02.mp4`）。  
> 注：旧文档曾写 `data/input/`，实际工作目录已统一为 `input_video/`。`data/input/` 仅作占位说明。

### 4.2 执行（三种入口）

```bash
# 方式 A: CLI 入口（推荐）
python -m redlight.app.cli --video input_video/违章02.mp4 --output data/output/run_02 --preset balanced

# 方式 B: 便捷脚本（等价于 A）
python scripts/run_video.py input_video/违章02.mp4 data/output/run_02 --preset balanced

# 方式 C: 诊断（只输出灯态/掩膜/占道时序，不写视频，便于快速验证）
python scripts/run_diag_generic.py input_video/违章02.mp4 data/output/diag_02.csv balanced
```

`--preset` 可选：`strict` / `balanced` / `loose` / `very_loose`（灵敏度递增，阈值见 `src/redlight/pipeline/tracker.py` 的 `SENSITIVITY_PRESETS`）。  
默认 `balanced`（speed=30px/s, sustain=5, duration=5, overlap=0.20）。

---

## 5. 输出说明

`--output` 目录下：
- `annotated.mp4`：带标注视频（斑马线青色填充、车辆框+ID、STOP 标黄、疑似/违规标红、左上角灯色、右下角 HUD）。
- `violations.csv`：事件列表，字段：
  `event_id, track_id, status, start_ts, end_ts, vehicle_class, confidence, light_state, signal_assumption, plate, evidence_image`
  - `status=confirmed`：**行人绿灯/闪烁 + 静止 + 压斑马线** 三条件同时满足且持续 `duration` 帧。
  - `status=review`：压线 + 静止，但**灯态未知（画面没拍到灯）且斑马线疑似被遮挡**，需人工复核。
  - `light_state`：`green` / `flashing` / `red` / `unknown`（语义见 §2 与设计文档 §5.3.1）。
  - `signal_assumption`：本工程假设识别对象 = **斑马线行人信号灯**（见设计文档 Q5），便于复核。
  - `plate`：该 track 的全局最佳车牌（多帧投票，见 §6 车牌说明）。
- `evidence/`：每起事件的证据截图（文件名含 event_id / track_id / 车牌）。

> **COT 小作文（规划中）**：按 2026-07-11 需求，后续每视频额外输出一份 `.md` 文字说明（灯态含遮挡推论 → 哪台车占道及比例 → 车牌读取时刻 → 最终结论），配合证据截图，便于定位"哪一步识别出错"。当前由 `Task #9` 跟踪，尚未接入主流程。

---

## 6. 参数调优（改 `configs/config.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `inference.fps` | 车辆检测采样帧率 | CPU 慢可降到 5；想要更稳升到 10 |
| `inference.imgsz` | 推理尺寸 | CPU 卡可 416/320，精度换速度 |
| `crosswalk.overlap_ratio` | 压线判定阈值(参考/可视化) | 误报多→调高(0.4)；漏报多→调低(0.2) |
| `crosswalk.cv_min_area` | CV 兜底最小条纹面积 | 场景尺度不同要改 |
| `traffic_light.*` | 红绿灯 v5 检测参数（见文件内注释） | 亮斑 `value_min`/`sat_min`、位置先验 `pedestrian_*`、`flicker_toggle_count` 等 |
| `violation.min_event_gap_sec` | 同 track 两次事件最小间隔(去重) | 默认 5 |
| `output.signal_assumption` | 信号类型假设(行人灯) | 报告标注用，一般不动 |
| **灵敏度预设** | 静止/压线/持续阈值 | **不在此配置文件**，由 `tracker.SENSITIVITY_PRESETS` 定义（`strict/balanced/loose/very_loose`），经 `--preset` 选择 |

> 注：旧文档把静止速度/持续/压线阈值写成 `stationary.speed_px_per_sec`、`stationary.sustain_frames`、`violation.duration_frames` 等独立配置键 —— **这些键已不存在**，相关逻辑统一收口到 `tracker.SENSITIVITY_PRESETS`，请勿再单独配置。

---

## 7. 提升精度（推荐路线 + 能力解耦评测）

当前开箱即用版本对"车辆检测"精度很高，瓶颈通常在**斑马线**和**红绿灯**两个模块。  
按 2026-07-11 的"能力解耦"要求，优化分三条独立线，各自评测，最后再综合：

1. **红绿灯识别**（最关键环节）：当前为 CV 颜色兜底 v5（亮斑 + 均值色分类，接住暗淡去饱和绿灯，修复 E16）。
   若要更稳，可用 LISA / GTSDB 训练 **YOLOv8n 二分类（red/green，不含 yellow）** 替换 `method: model`，把权重放到 `models/` 并在 `configs/config.yaml` 指向它（`method` 保持 `auto`/`model` 即可自动启用）。
2. **斑马线检测**：当前为经典 CV 兜底 **v11**（多位置条带扫描 + 车辆锚定，修复 E17 泛化失败）。
   若要更稳，用 [CDSet-3434](https://zenodo.org/records/8289874) 或 Roboflow `crosswalk_seg` 训练 `yolov8n-seg`，得到 `crosswalk_seg.pt` 放到 `models/`。
3. **车牌识别**：HyperLPR3 (HIGH) + `PlateConsensus` 多帧加权投票。**车牌是全局读取的** —— 不一定在违章帧，可能在视频前半或后半才看清，按 `track_id` 关联占道车（见设计文档 E19）。

**模块化评测（进行中）**：`datasets/gt/events.csv` 为权威真值（事件级时段标注）；`scripts/eval_light_all.py` 已能跑红绿灯状态的 per-frame 准确率/宏 F1；最终目标把评测拆成 **红绿灯 / 静止占道 / 车牌 OCR** 三项独立指标 + 端到端事件级 P/R/F1（见设计文档 §6 与 `Task #10`）。

---

## 8. 已知限制与对策

| 限制 | 对策 |
|---|---|
| 合法停在**停止线前**（斑马线外侧）被误判 | 精确斑马线掩膜 + footprint IoU 阈值（`footprint=0.5` 只取车体下半部足迹，去除 YOLO 大框对车顶/天空的稀释，E15/E17 实证） |
| 红灯前已进入、正在清空的车 | 本工程**红灯不判违规**（语义反转，见 §2）；仅绿灯/闪烁 + 静止 + 压线才违规 |
| 移动机位下斑马线 CV 兜底不稳定 | 优先提供 `crosswalk_seg.pt` 分割模型（v11 已用多位置扫描缓解） |
| 画面拍不到红绿灯 | 自动降级为 `unknown`，事件进"待复核"，不自动判定 |
| 红车/反光导致颜色兜底误判 | v5 用亮斑+均值色分类 + 位置先验（路面车灯区 `ped=0` 直接剔除）已大幅缓解（E16）；仍不稳则升级小模型 |
| 暗淡/去饱和的 LED 绿灯看不见 | v5 不再卡极端饱和像素，改用亮斑掩膜 + bbox 内 HSV 均值分类（E16 修复） |

---

## 9. 工程结构

```
redlight-crosswalk-violation/
├── configs/config.yaml          # 全部可调参数(含详细注释)
├── requirements.txt
├── input_video/                 # 待检测视频(违章01~11.mp4) + label_result_01.csv(GT)
├── datasets/gt/                 # 真值: events.csv(事件级) + light_state/ + violation_events/
├── scripts/                     # 工具/诊断脚本(run_video / run_diag_generic / eval_light_all ...)
├── models/                      # 模型权重(可选, 缺省走 CV 兜底)
├── src/redlight/                # 源码(分层包)
│   ├── infrastructure/          # L1: config / 几何(IoU, footprint overlap)
│   ├── data_pipeline/           # L2: frame_sampler
│   ├── models/                  # L3: vehicle(YOLOv8n) / crosswalk(v11) / traffic_light(v5) / plate(HyperLPR3)
│   ├── inference/               # L4a: engine
│   ├── evaluation/              # L4c: metrics / evaluator
│   ├── pipeline/                # L5: dag / tracker(V2) / violation_engine(V2) / plate_consensus / visualizer
│   └── app/                     # L6: cli
└── data/output/                 # 运行输出
```

---

## 10. 下一步

把你的手机视频放到 `input_video/`，我们一起：
1. 跑 `run_video.py` 看 baseline 效果；
2. 用 `datasets/gt/events.csv` 真值量化红绿灯/占道/车牌三项能力（模块化评测）；
3. 若某单项不准，按第 7 节独立微调对应模型；
4. 接入 COT 小作文 + 截图输出（规划中）。
