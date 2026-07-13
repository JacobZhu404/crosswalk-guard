# 分层架构细化设计 — 感知 / 时序 / 判定 / 呈现 / 标注

> 状态:草案 v1(Claude Code 起草,2026-07-13)。三方(CC / Lingma / CC2)就四条洞察已对齐,本文将其固化为**唯一权威架构 spec**,建立在 `docs/plans/2026-07-12-design-requirements-v2.md`(需求 v2,§3-4 已有"中间态 JSON→判定层+COT层"雏形)之上。
> 语义基线:违规 = 行人绿灯/闪烁 + 车静止 + 压斑马线(E12)。
> **协作铁律:先 spec 后实现,不并行重构同一模块**(共享工作树已栽两次)。

---

## 1. 动机(为什么重构)

当前时序逻辑**散落**,且红绿灯检测器是"有状态怪物":
- 灯态平滑(`smoothing_window`/`hysteresis`)+ 空间锚 + 轨迹持久计数 **全塞在 `TrafficLightDetector` 内部** → 单测必须喂帧序列、实例带状态不可并行、颜色识别与时序/空间锁定混在一起。
- 车牌投票在 `PlateConsensus`、静止在 `TrackStateManagerV2`、向前填充在 `violation_engine._resolve_light`、跟踪在 `vehicle.SimpleTracker` —— 四处各管一段时序。
- 判定引擎为"逐帧累积状态机 + unknown 向前填充 hack",复杂且难解释。

目标:**检测器退化为纯逐帧感知;所有时序推理收口到一层;判定退化为区间代数;呈现/标注消费同一中间态。**

## 2. 目标分层

```
① 逐帧感知 (无状态, 静态图可调可评)
     frame -> raw_detection (+ per-frame confidence)
     车辆框 / 斑马线掩膜 / 车牌OCR / 灯候选+每帧原始色·标签
② 时序后处理 (统一 TemporalFusion)
     raw序列 + 轨迹 -> 结构化中间态(时间线)
     - 时间不变属性: 全局投票/回填 (车牌·车种·track身份)
     - 时变状态: 分段+有界填充+置信融合 (灯色·静止·占道)
③ 判定 (最后, 区间交集代数)
     绿段 ∩ 静止区间 ∩ 压线区间 ≥ duration -> confirmed/review/none
④ 呈现 (消费中间态): 标注视频 / COT小作文 / 截图
⑤ 横切: 通用画廊人工标注 (作用于 ①②③ 任意产物 vs GT)
```

## 3. 层接口与数据流

**帧源(DAG 独立节点, A-D4)**:统一 `FrameSource`(封装 `FrameDataset` 预抽帧 / `VideoSampler` 视频),各检测器从同一节点取帧、各自声明采样频率;调 fps/分辨率只改一处。

**① 检测器(纯函数, A-D1)**:`detect(frame) -> RawDetection`,无内部时序状态,附 `confidence`。
- 车辆:`[{box, cls, conf}]`(去掉 tracker,tracker 归②)
- 斑马线:`mask`(+ 质量分)
- 车牌:`[{box, text, conf}]`(逐帧 OCR,不投票)
- 红绿灯:`{candidates:[{box,cx,cy}], per_frame_obs: green|red|off|None, conf}`(**不做窗口投票/锚点/闪烁判定**——全归②)

**② TemporalFusion(统一)**:输入各检测器的逐帧 raw 序列 + 帧时间戳,输出中间态(§5)。
- 轨迹关联(IoU 贪婪,复用 `SimpleTracker` 逻辑搬入本层)。
- 时间不变属性:按 track 全局加权投票(车牌复用 `PlateConsensus` 算法)。
- 时变状态:先按 GT/自身一致性分段,再段内滑窗投票 + 迟滞 + 置信融合(灯态搬 `smoothing_window`/`hysteresis`);静止判定搬 `TrackStateManagerV2`;占道逐帧算 overlap 后归入区间。

**③ 判定(区间代数, A-D3)**:消费中间态,`违规区间 = 交集(绿/闪段, 静止区间, 压线区间)`,长度 ≥ duration → confirmed;灯未知+遮挡段 → review;否则 none。无流式状态机。

## 4. 关键决策

| 编号 | 决策 | 理由 |
|------|------|------|
| **A-D1** | 检测器**无状态**,仅出单帧 raw + confidence | 可单测(喂单图)、可并行、职责单一;使①的静态图调参/评测通用化(Q1)真正成立 |
| **A-D2** | **时间不变 vs 时变** 分治(Q3 命门) | 时间不变(车牌/车种/身份)可全局投票+任意前后回填;**时变(灯色/静止/占道)只能同质段内有界填充,严禁跨真实跳变**(否则绿段边界把红填成绿→判反) |
| **A-D3** | 判定 = **区间交集代数**,放最后 | 离线批处理无实时约束;规则简单、可解释、好测;等②全局推理完成再判(Q4) |
| **A-D4** | 帧源统一为 DAG 节点 | fps/分辨率单点调参,去各检测器自管采样(Q1) |
| **A-D5** | 画廊抽象为通用 `ReviewGallery`/`BaseGalleryBuilder` | `(item, pred, gt, mismatch_only) -> 网格/裁图 + 交互标注 + 导出 labels`;服务灯态/车牌/M1 crop/未来违规帧复核(Q2)。**保持轻量,勿造标注平台** |

## 5. 中间态 JSON schema(②产出,③④消费;细化 design v2 §4.1)

```json
{
  "video": "违章02", "fps": 8.0, "duration_s": 110.4,
  "light_segments": [   // 时变: 已分段+融合后的灯态时间线
    {"start_s":0,"end_s":20.9,"state":"red","conf":0.9,"evidence":"visible"},
    {"start_s":20.9,"end_s":85,"state":"green","conf":0.85,"evidence":"visible"}
  ],
  "tracks": [{
    "track_id":3, "vehicle_class":"car",         // 时间不变: 全局投票
    "plate":{"text":"京LNE560","conf":0.91,"read_at_s":[95.2]},  // 时间不变
    "stationary_intervals":[[21,68]],            // 时变: 静止区间
    "occupancy_intervals":[{"start_s":21,"end_s":68,"max_overlap":0.42}]  // 时变
  }],
  "review_flags":["..."]
}
```
判定层只需对 `light_segments`(绿/闪) × `tracks[].stationary_intervals` × `occupancy_intervals` 做区间交集。

## 6. 现有模块 → 目标层映射 + 拆分清单

| 目标层 | 现有 | 改动 |
|--------|------|------|
| ① 感知 | `vehicle/crosswalk/plate/traffic_light.py` | **红绿灯拆时序(最重)**;车辆去 tracker;其余基本就绪 |
| ② 时序 | `PlateConsensus`✅ / `tracker.py` / `analysis.py`雏形 | **新增 `TemporalFusion`**,收敛灯态平滑、静止、向前填充、轨迹关联 |
| ③ 判定 | `violation_engine.py` | 逐帧状态机 → 区间交集代数 |
| ④ 呈现 | `visualizer.py` + COT(待建) | 消费中间态 JSON |
| ⑤ 标注 | `make_light_gallery/make_plate_gallery.py` | 抽象 `BaseGalleryBuilder`(Lingma 之前已规划) |

**`traffic_light.py` 待搬出到②的具体清单**(拆分依据):
- 状态字段:`global_recent`(窗口投票)、`hysteresis`/`_last_state`(迟滞)、`heads`/`tracks`/`anchor` + `anchor_hold`/`reanchor_need`(空间锚持久)、`_fi`(帧计数)、`signal_prior`+`prior_hold` 相关。
- 逻辑:`_state_from_global`(窗口多数决/闪烁判定)、`_select_lit` 里的锚点迟滞/重锚、`_resolve_light`(在 violation_engine,向前填充)。
- **留在①的**:`_candidates`(HSV 亮斑)、`_classify`(单帧色)、M1 `signal_state_classifier`(单帧 walk/stand/off)。即"这帧看到什么"留①,"跨帧怎么定状态"去②。
> ⚠️ 空间锚有双重性:它既是"定位稳定"(空间)又被用作"状态保持"(时序)。拆分时:**定位候选留①,锚点的跨帧保持/迟滞归②**。

## 7. 基建现状(顺风,可复用)
- `FrameDataset`(预抽帧+manifest)、`VideoSampler`(视频采样)、`GTLookup`(GT 段/车牌加载 + `state_at`)、`image_utils`(save_jpg/robust_imread/conf色)、`report_formatter`(Evaluator 薄封装,尚未接线)、`metrics`(含 mask_iou/MOTA/light_state/event 全套)、`PlateConsensus`(②的时间不变投票范式)。

## 8. 推进顺序(按技术债 × 收益)
1. **② TemporalFusion + 中间态 JSON 定型**(收益最大,红绿灯拆时序是核心);M1 分类器作为①的灯态感知拼图**可并行独立推进**,产出后其时序并入②是第一个自然步骤。
2. **③ 判定改区间代数**(依赖②的中间态)。
3. **④ COT** 消费中间态(design v2 §4)。
4. **⑤ 画廊通用化** + **帧源节点(A-D4,低成本早收益,可先做)**。

## 9. 协作分工(草案,待认领)
- **本 spec** = 唯一权威;实现前在此对齐,勿并行改同一模块。
- CC2 认领补充:`FrameSource`/`GTLookup` 基建能力细目、判定层区间代数细节。
- Lingma 认领:`BaseGalleryBuilder`(⑤)、eval 基建延续。
- CC(我):② `TemporalFusion` 接口 + 红绿灯时序拆分、中间态 schema、M1 收尾。
- 每人各用独立 worktree/clone;scoped `git add`;trunk-based 提 main。

## 10. 风险 / 开放项
- 分段边界:时变状态填充依赖"同质段"划分,分段本身若错会连锁误判 → 分段用高置信帧锚定,低置信不参与定段。
- 长视频中间态内存:8fps 几千帧,可接受。
- 重构期回归:每搬一块必保 `run_tl_tests` + `pytest tests/` 绿;②上线前用现有 `eval_light_all`(prior-free)对齐新旧灯态输出。
- 大幅运镜稳掩膜(design v2 D5 遗留)不在本 spec。
</content>
