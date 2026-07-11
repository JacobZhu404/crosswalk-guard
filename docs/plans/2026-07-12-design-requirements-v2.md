# 设计需求 v2 — 红灯压斑马线违规检测（行人绿灯相位车辆静止占道）

> 状态：已与用户 grill-me 对齐（2026-07-12），作为**唯一权威需求规格**。
> 关联文档：`2026-07-10-architecture-redesign-design.md`（架构设计 / how），本文档为需求 / what+why。
> 语义基线：违规 = **行人绿灯 / 闪烁清空相位 + 车辆静止 + 压在斑马线上**（E12 反转，红灯=车可通行不违规）。

---

## 1. 四条核心需求（用户原话提炼）

**R1 — 可解释输出（COT / VQA 式）**
输出不能只是端到端的"有无违规"。必须附带一段**类思维链（COT）小作文 + 对应截图**，让人能逐环定位"是哪一层出错"。
示例格式（强制包含要素）：
> "X分Y秒 – X分Z秒，行人灯是绿色；X分W秒是红色；X分V秒–X分U秒无红绿灯（被遮挡），推测为绿灯。
> 视频中第 N 台车（track_id=xxx）从 xx 秒到 xx 秒占据斑马线 30%；符合违章定义。
> 其车牌号为 京Axxxx，在 xx 秒清晰可见……"

**R2 — 六模块能力 + 双层输出**
识别系统必须包含以下独立模块：
1. 红绿灯识别 **+ 推理**（含遮挡→推测、未知短时向前填充）
2. 车辆识别
3. 斑马线识别
4. 斑马线占用比例计算
5. 车牌号识别
6. 车辆跟踪（同一台车可能移动；拍摄者也可能移动）

最终产出分两层：
- **判定层（规则模型）**：确定性、可复现，输出 violations.csv + 结构化中间态 JSON。
- **呈现层（COT 小作文）**：把结构化中间态渲染成自然语言 + 截图，供人 review / 组织。**v1 用确定性模板，不引入 LLM/VQA**（见 D3）；LLM 润色留作未来可选。

**R3 — 每层/每模型可独立评测**
每个模块都能单独跑：准确率/召回率（P/R/F1）、mask（斑马线）、真值（GT）对比、正负例标注。要求评测与训练/推理解耦（要求#9）。

**R4 — 需求合理性复核**
用户要求 AI 主动指出需求中不合理/有内部张力的地方并沟通（即本 gril-me 过程）。

---

## 2. 已锁定的设计决策（grill-me 结论）

| 编号 | 决策点 | 结论 | 理由 |
|------|--------|------|------|
| **D1** | 遮挡时灯态处理 | COT 叙述"推测绿灯"，但 **verdict 标 `review`** 交人复核 | 调和"偏向多报(Q1)"与"安全侧不自动 confirmed 违规"；避免把红灯误推成绿灯直接判违规 |
| **D2** | 占用比例分母 | **(车∩mask) / mask 面积** = "斑马线被车覆盖的比例" | COT"占据斑马线 30%"读着最自然；需改 `compute_overlap_ratio` 分母（当前是 /车框面积） |
| **D3** | COT 实现方式 | **v1 确定性模板渲染**，LLM/VQA 留未来 | 零外部依赖、CPU-only、可复现，符合"避免 VLM"偏好 |
| **D4** | 模块化评测推进 | 先补 GT + 跑通灯态/事件评测；车辆/车牌 GT 用户逐步补；**新增 mask-IoU 与 MOTA 指标函数** | 评测脚手架已覆盖 4/6 模块，瓶颈是真值数据 |
| **D5** | 拍摄方式假设 | **基本固定机位 / 小幅手持抖动**（v1 无需帧配准） | v11 逐帧重检即可；大幅运镜导致的掩膜错位（E17）留作未来 stabilization 工作 |
| **D6** | 规格落盘 | 写本文档为唯一权威需求规格 | 防止 grill-me 中需求被 miss |

---

## 3. 系统模块架构与数据流

```
视频帧
  ├─[M6 跟踪]── SimpleTracker(IoU 贪婪) → track_id 稳定(车移动/拍摄者移动均保 ID)
  ├─[M2 车辆]── YOLOv8n → 车框 + vehicle_class
  ├─[M3 斑马线]─ CrosswalkDetector v11(多位置扫描+车辆锚定) → mask
  ├─[M4 占用]── compute_overlap_ratio(box, mask, footprint=0.5, denom=mask) → 占用%
  ├─[M1 红绿灯]─ TrafficLightDetector v5(亮斑掩膜+ bbox 均值色) → 灯态
  │     └─[推理] _resolve_light: 未知短时向前填充 + 遮挡→推测绿灯(标 review)
  ├─[M5 车牌]── HyperLPR3 + plate_consensus(全局按 track 聚合) → 每 track 车牌
  └─[规则引擎]── 三条件状态机: 灯态∈{绿,闪} ∧ 静止 ∧ 占用%≥T → confirmed/review/none
        ├─ violations.csv (规则 verdict)
        ├─ <video>_analysis.json (结构化中间态)
        └─[COT 渲染器]── COT_<video>.md + screenshots/  (D3 模板)
```

**关键不变式**
- 车牌读取**视频全局**（不一定在违章帧），按 track_id 关联占道车（E19）。
- 斑马线掩膜质量门：面积/可见度过低 → 该帧标 low-confidence，COT 显式标注（D5 兜底）。
- 占用阈值 T 沿用 0.15–0.25 区间，切到 mask 分母（D2）后需在 02/03/04 重新标定基线。

---

## 4. COT 小作文规范

### 4.1 结构化中间态 JSON（COT 渲染器的输入，也是模块化评测的载体）
```json
{
  "video": "违章02", "duration_s": 110.4,
  "segments": [
    {"start_s": 0,  "end_s": 21, "light_state": "red",    "conf": 0.9, "occluded": false, "reason": "..."},
    {"start_s": 21, "end_s": 68, "light_state": "green",  "conf": 0.85,"occluded": false, "reason": "..."},
    {"start_s": 69, "end_s": 76, "light_state": "red",    "conf": 0.8, "occluded": false, "reason": "车移动"},
    {"start_s": 79, "end_s": 104,"light_state": "green",  "conf": 0.7, "occluded": true,  "reason": "被遮挡推测绿灯→review"}
  ],
  "vehicles": [
    {"track_id": 3, "vehicle_class": "car", "plate": "京LNE560",
     "plate_read_at_s": [95.2, 96.0], "plate_conf": 0.91,
     "stationary": true,
     "occupancy": [{"start_s": 21, "end_s": 68, "max_overlap": 0.42, "denom": "mask"}],
     "verdict": "confirmed_violation"}
  ],
  "events": [{"track_id": 3, "status": "confirmed_violation", "start_s": 21, "end_s": 68,
              "light_state": "green", "max_overlap": 0.42, "plate": "京LNE560"}],
  "review_flags": ["segment@79-104: 灯被遮挡推测绿灯，置信低需复核"]
}
```

### 4.2 COT .md 模板（强制要素，对应 R1 示例）
```
# 视频 {video} 解析

## 红绿灯时间线
- {start}–{end}：行人灯 **{green/red/flashing/unknown}**（置信 {conf}）{若遮挡："（被遮挡，推测为绿灯→review）"}
- …

## 占道车辆
- 第 {N} 台车（track_id={id}，{vehicle_class}）：{start}–{end} 占据斑马线 **{pct}%**（分母=斑马线面积）。
  {stationary? "静止" : "移动"} → {符合/不符合} 违章定义。
- 车牌号 **{plate}**，在 {plate_read_at_s} 秒清晰可见（置信 {plate_conf}）。

## 结论
- track_id={id}：**{confirmed_violation / review / none}**。

## 存疑 / 需复核
- {review_flags}
```

### 4.3 截图策略（每视频）
| 帧类型 | 选取规则 | 用途 |
|--------|----------|------|
| 灯态证据帧 | 绿/红/遮挡/闪烁各取 1 帧 | 佐证灯态判定 |
| 占用峰值帧 | overlap 最大那帧，叠加车框+掩膜 | 佐证占道比例 |
| 车牌清晰帧 | 该车 track 上 OCR 置信最高帧 | 佐证车牌读取 |

输出目录：`data/output/cot/{video}/`（COT_<video>.md + screenshots/）。

---

## 5. 模块化评测规范（R3）

### 5.1 各模块评测现状与缺口
| 模块 | 指标函数现状 | GT 需求 | 状态 |
|------|--------------|---------|------|
| M1 红绿灯 | ✅ `light_state_metrics`（P/R/F1/逐类） | per-frame 灯态 GT（events.csv 可展开） | 可立即跑 |
| M2 车辆 | ✅ `detection_metrics` + mAP | per-frame 车框 GT | 待补 GT |
| M3 斑马线 | ❌ 缺 `mask_iou` | per-frame mask GT | **需写指标+补 GT** |
| M4 占用比例 | 复用 M3 mask + M2 框 | 占用% 样本 GT | 待补 GT |
| M5 车牌 | ✅ `plate_accuracy`（exact/char/省份） | per-track 车牌 GT | 待补 GT |
| M6 跟踪 | ❌ 缺 `MOTA`/ID-switch | per-frame track_id GT | **需写指标+补 GT** |
| 端到端事件 | ✅ `event_metrics`（T-IoU P/R/F1） | 事件级 GT（events.csv 已有） | 可立即跑 |

### 5.2 验收阈值（建议，可复议）
| 维度 | 指标 | 阈值 |
|------|------|------|
| 端到端事件 | recall / FP | recall≥0.9；01/10 零误报 |
| M1 红绿灯 | 逐类 P/R | ≥0.8（绿/红/闪/未知） |
| M3 斑马线 | mask-IoU | ≥0.7 |
| M4 占用 | 绝对误差 | \|pred−gt\|≤0.05 |
| M2 车辆 | mAP@0.5 | ≥0.5 |
| M5 车牌 | exact / char | ≥0.8 / ≥0.95 |
| M6 跟踪 | MOTA | ≥0.7 |

### 5.3 GT 推进策略（D4）
1. **立即**：events.csv 展开 per-frame 灯态 GT + 事件 GT → 跑通 M1/M端到端评测。
2. **用户补**：车辆框 / 车牌 /（可选）掩膜 / track_id 真值，逐步解锁 M2/M4/M5/M3/M6。
3. **代码补**：`mask_iou` 指标函数（M3）、`MOTA` 指标函数（M6）。

---

## 6. 当前状态与现实检查（必须正视）

- **M1 红绿灯 v5 实测（全 11 视频，pred vs events.csv 展开 GT）**：`accuracy=0.36 / macro-F1=0.26`；其中 **07 / 11 / 01 接近 0**（单独 probe 确认这些视频检测器几乎找不到信号灯）。
- **含义**：底层灯态识别仍是 36% 水平时，COT 再漂亮也是"垃圾进垃圾出"。**R1/R3 的前提（单能力准）尚未满足**。
- **待办根因**：07/11/01 失败模式需逐个诊断（信号是否不在顶部 / 颜色-亮度分布不同 / 画面构图差异）。

---

## 7. 推荐推进顺序（分阶段，呼应"先保证单能力准"）

1. **修 M1 灯态识别**：诊断 07/11/01 失败模式并修复（多位置/颜色分布泛化）。
2. **切 D2 分母 + 标定 T**：`compute_overlap_ratio` 分母改 mask 面积；在 02/03/04 标定占用阈值 T（0.15–0.25 区间重新基线）。
3. **COT 模板 + 截图**（D3）：渲染器消费 §4.1 JSON → §4.2 .md + §4.3 截图。
4. **模块化评测**（D4）：灯态/事件先跑；补 `mask_iou`/`MOTA`；车辆/车牌 GT 就位后补评。

---

## 8. 开放项 / 未来工作
- 大幅运镜视频 → 帧配准(stabilization) 稳掩膜（D5 当前假设不成立时启用）。
- LLM/VQA 润色 COT 叙事（D3 未来层）。
- 斑马线 mask GT 采集（半自动弱标签起步可选）。

---

## 9. 经验教训索引（关联 MEMORY.md E1–E19）
- E12 语义反转、E13 双模式 footgun、E14 遮挡判定、E15 草地误检、E16 暗淡绿灯、E17 掩膜错位、E18 COT 可解释、E19 车牌全局聚合。
- 本轮新增约束：D1 遮挡→review、D2 占用分母=mask、D3 COT 模板非 LLM、D5 固定机位假设。
