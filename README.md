# 红灯停车压斑马线检测工程

纯 CV 方案（**不依赖 VLM 大模型**），面向**本地 Windows + CPU** 批量处理手机录制视频。
目标：检测"红灯期间，车辆停在斑马线（斑马线）上"的违规行为，输出标注视频 + 违规事件报告 + 证据截图。

---

## 1. 为什么不用 VLM

- 你的场景本质是**几何 + 状态机判定**（红灯 ∧ 车静止 ∧ 车压斑马线），用检测 + 跟踪 + 规则即可稳定、可复现地解决。
- VLM 推理慢、结果有随机性、成本高，不适合做"判定"这种硬规则任务。
- 本方案全部为确定性 CV 推理，CPU 可跑，结果稳定可解释。

---

## 2. 技术架构

```
视频帧
  ├─ 车辆检测     YOLO11n (COCO, car/bus/truck) + ByteTrack 跟踪
  ├─ 斑马线检测   分割模型(可选) / 经典CV兜底(自适应阈值+条纹轮廓)
  ├─ 红绿灯检测   状态模型(可选) / 颜色兜底(HSV亮斑投票) + 时间平滑
  ├─ 静止判定     跟踪质心速度 < 阈值 且 持续 N 帧
  └─ 违规状态机   红灯 ∧ 静止 ∧ 压斑马线 且 持续 duration 帧
        ↓
  输出: annotated.mp4 + violations.csv + evidence/*.jpg
```

**关键设计（针对你的约束）：**
- **CPU-only** → 全用 nano 级模型 + 降采样推理（默认 8fps 足够抓静止车）。
- **机位不固定** → 斑马线/红绿灯逐帧检测，移动拍摄也能 work。
- **部分路段拍不到红绿灯** → 红灯状态返回 `unknown`，相关疑似事件归入 **待复核 (status=review)**，绝不瞎判。

---

## 3. 环境安装

需要本机有 Python 3.9~3.12（**当前沙箱无 Python，请在你本机执行**）。

```bash
cd redlight-crosswalk-violation
pip install -r requirements.txt
```

可选：Intel CPU 加速（提速明显）
```bash
pip install openvino openvino-dev
# 然后把 yolo11n.pt 导出为 OpenVINO: yolo export model=yolo11n.pt format=openvino
```

---

## 4. 运行

### 4.1 准备模型
```bash
python scripts/download_models.py
```
- 车辆模型 `yolo11n.pt` 会自动下载。
- 斑马线分割 / 红绿灯状态模型**可选**：不提供权重时自动用 CV/颜色兜底，工程也能跑通。

### 4.2 放入视频
把手机视频放到 `data/input/`（如 `data/input/sample.mp4`）。

### 4.3 执行
```bash
# 用默认配置
python -m src.pipeline

# 指定视频/输出
python -m src.pipeline --video data/input/sample.mp4 --output data/output
```

---

## 5. 输出说明

`data/output/` 下：
- `annotated.mp4`：带标注视频（斑马线青色填充、车辆框+ID、STOP 标黄、疑似/违规标红、左上角灯色、右下角 HUD）。
- `violations.csv`：事件列表，字段：
  `event_id, track_id, status, start_ts, end_ts, vehicle_class, confidence, red_light, evidence_image`
  - `status=confirmed`：红灯确认 + 压线 + 静止。
  - `status=review`：压线 + 静止，但**红灯状态未知**（画面没拍到灯），需人工复核。
- `evidence/`：每起事件的证据截图。

---

## 6. 参数调优（改 `configs/config.yaml`）

| 参数 | 作用 | 调参建议 |
|---|---|---|
| `inference.fps` | 车辆检测采样帧率 | CPU 慢可降到 5；想要更稳升到 10 |
| `inference.imgsz` | 推理尺寸 | CPU 卡可 416/320，精度换速度 |
| `crosswalk.overlap_ratio` | 压线判定阈值 | 误报多→调高(0.4)；漏报多→调低(0.2) |
| `stationary.speed_px_per_sec` | 静止速度阈值 | 画面尺度不同要改；可看 HUD 的 STOP 数 |
| `stationary.sustain_frames` | 持续多少帧算静止 | 默认 8（@8fps≈1s） |
| `violation.duration_frames` | 持续多久确认违规 | 过滤瞬时通过；默认 8 |
| `traffic_light.smoothing_window` | 灯色平滑 | 闪烁误判就加大 |

---

## 7. 提升精度（推荐路线）

当前开箱即用版本对"车辆检测"精度很高，瓶颈通常在**斑马线**和**红绿灯**两个模块。
若要更稳，建议自行微调两个小模型（数据量不用大，几百张即可）：

1. **斑马线分割**：用 [CDSet-3434](https://zenodo.org/records/8289874)（3434 张车载视角，含白天/雨天/夜晚/遮挡）或 Roboflow `crosswalk_seg` 训练 `yolo11n-seg`，得到 `crosswalk_seg.pt`。
2. **红绿灯状态**：用 LISA / GTSDB 训练 `yolo11n` 三分类（red/green/yellow），得到 `traffic_light.pt`。
3. 把权重放到 `models/` 并在 `configs/config.yaml` 指向它们，`method` 保持 `auto` 即可自动启用。

---

## 8. 已知限制与对策

| 限制 | 对策 |
|---|---|
| 合法停在**停止线前**（斑马线外侧）被误判 | 精确斑马线多边形 + overlap 阈值；停止线在斑马线边缘外侧，合法车不应落入条纹区 |
| 红灯前已进入、正在清空的车 | v1 用"红灯期间持续静止在斑马线上"定义；必要时加"进入时刻"追踪排除合法清空 |
| 移动机位下斑马线 CV 兜底不稳定 | 优先提供 `crosswalk_seg.pt` 分割模型 |
| 画面拍不到红绿灯 | 自动降级为 `unknown`，事件进"待复核"，不自动判定 |
| 红车/反光导致颜色兜底误判红灯 | 用 `traffic_light.pt` 模型替代颜色兜底 |

---

## 9. 工程结构

```
redlight-crosswalk-violation/
├── configs/config.yaml          # 全部可调参数
├── requirements.txt
├── scripts/download_models.py   # 模型准备
├── src/
│   ├── utils.py                 # 配置加载 + 几何计算
│   ├── vehicle_detector.py      # YOLO11n + ByteTrack
│   ├── crosswalk_detector.py    # 分割 + CV 兜底
│   ├── traffic_light.py         # 模型 + 颜色兜底 + 平滑
│   ├── tracker.py               # 静止判定
│   ├── violation_engine.py      # 违规状态机
│   ├── visualizer.py            # 可视化
│   └── pipeline.py              # 主流程
├── data/input/                  # 放待检测视频
└── data/output/                 # 输出结果
```

---

## 10. 下一步

把你的手机视频发我（或放到 `data/input/`），我们一起：
1. 跑一版看 baseline 效果；
2. 根据实际画面调 `configs/config.yaml` 的阈值；
3. 若斑马线/红绿灯识别不准，按第 7 节微调两个小模型。
