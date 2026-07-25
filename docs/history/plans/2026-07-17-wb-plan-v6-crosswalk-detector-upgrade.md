# Plan v6 — 斑马线检测器升级（占道最高杠杆）

日期：2026-07-17 | 作者：wb | 状态：交 cc 复核（含 Jacob 拍板点）
依据：docs/handoff/2026-07-17-cc-direction-crosswalk-detector-upgrade.md + 天花板诊断 diag_gt_crosswalk_ceiling.py

## 0. 目标与已锤实依据

终点：端到端 F1 从 **0.636 → ~0.94**（只剩 04 灯态），靠「修斑马线检测」一项。**注：此为 Phase 2 seg 完全态的终点；Phase 1 train-free 探针仅验证通路 + 根除 flicker，预期 mask-IoU~0.4，不达 bar（见改定 1）。**
天花板诊断（GT 多边形当 mask 塞管线，9 违章视频）：
| 配置 | P | R | F1 | 覆盖 |
|---|---|---|---|---|
| 基线 v11 | 0.538 | 0.778 | 0.636 | 0.43 |
| GT+denom=mask | 1.0 | 0.778 | 0.875 | 0.92 |
| GT+denom=box | 1.0 | 0.889 | **0.941** | 0.90 |

- v11 mask-IoU 仅 **0.011**（全宽薄带 vs 真·透视多边形）→ 根因是「只能出水平带，出不了斜多边形」。
- denom=box 比 mask 多救 1 TP（11）。04 两种 denom 都漏 → 灯态绑定，出局。

## 1. 已重读代码核实的关键事实（落地锚点）

- **注入点已存在**：`cli.run(crosswalk_detector=, occ_denom=)`（cli.py:30-31, 75, 71-72）已支持替换检测器与占道分母。天花板诊断即由此验证，本计划复用，不新开注入机制。
- **新检测器接口**：只需实现 `detect(frame, vehicle_boxes=None) -> mask(h,w) uint8`（GtCrosswalkDetector 即样例）。`CrosswalkDetector` 在 `cli.run` 内**每视频新建**（cli.py:75），状态天然不跨视频。
- **denom=box 已落地**：`BatchViolationEngine(occ_denom=)` → `compute_overlap_ratio(box, mask, footprint=0.5, denom=occ_denom)`（violation_engine.py:179, geometry.py:23-53）。
- **阈值位置**：`SENSITIVITY_PRESETS` 的 `overlap`（tracker.py:15-24）：balanced=0.20 / strict=0.30 / loose=0.15。**此 0.20 是给 denom=mask 标的**（brief 确认）。
- **模块指标已就位**：`eval_crosswalk_mask.py` 算 mask-IoU vs GT 多边形（现状 0.011）。
- **GT 现状**：`datasets/gt/crosswalk/*.json` 9 视频共 **35 帧**多边形（够评测，不够从头训）。

## 2. ★ Q1 方案抉择 + 数据需求评估（诚实）

### 公开权重核实（2026-07-17 WebSearch）
- Roboflow Universe 有**多个 crosswalk 模型**，但绝大多数是 **bbox 目标检测**（非多边形 mask），且训练域为国外街道/航拍/特定场景 → **与中国监控式行人斑马线有域差**。
- SegFormer fine-tune on sidewalk 含 crosswalk 类，**crosswalk IoU 仅 0.5071**，且 transformer 在 CPU 重。
- YOLOv8-seg 支持实例分割 + 导出 ONNX（ultralytics），可**用我们的 GT 微调**。
- **结论：无现成「域适配」预训练权重可零样本直迁**。现实路径 = 在我们自己 GT 上微调，或用 train-free CV。

### 方案对比与推荐
| 方案 | 可行性 | 数据需求 | 风险 |
|---|---|---|---|
| (A) 分割模型直迁预训练 | ❌ 域差，零样本不行 | — | 误检/漏检 |
| (B) 改进CV出多边形（train-free） | ✅ 立即可做，CPU 友好 | **0 新标注** | 斜多边形精度上限不如 seg |
| (C) YOLOv8-seg 微调 | ✅ 可行 | 35 帧+增强 可起步；稳健需 +~100-200 帧 | 需训练/算力；过拟合 9 场景 |
| (D) 混合：B 过渡 + C 按需 | ✅✅ | B 0；C 视 B 结果 | 分期可控 |

**推荐：D（混合，train-free 探针先上 → seg 是真解）**

> ⚠️ **改定 1 — 诚实下调 Phase 1 预期（cc 复核要求）**
> 实测遮挡事实：违章11@21.7 仅标「可见条纹」vs GT 全多边形 = mask-IoU **0.404**，而约 **60% 的 GT 面积位于车底下**（需由未遮挡区域重建/外推）。
> **train-free CV 与 VLM 一样看不见车底下的斑马线** → Phase 1 CV 大概率卡在 **~0.4，达不到自身定的 ≥0.5 bar**。
> 因此：
> - **Phase 1 不是 "the fix"，是便宜探针**：0 标注、顺带根除 flicker、验证「注入点 + denom=box + 阈值扫描」通路；预期**会不达标**（mask-IoU~0.4），别让 Jacob 误判 "train-free 大概率够"。
> - **Phase 2（seg）才是真解**：分割模型学过斑马线形状先验，能「脑补」车底下被遮挡的延续段（CV/VLM 都做不到）。从计划原「rare fallback」升级为 **"probable next step"**——Phase 1 实测确认 ~0.4 即进入 seg。

- **Phase 1（train-free 探针）**：改进 CV，让检测器**直接输出透视四边形掩膜**（而非水平带），配合 Q2 时间聚合、Q3 denom=box。零新标注、CPU 可跑、立即用注入点验证管线。目标 = 验证通路 + 根除 flicker；**mask-IoU 不要求达 bar（预期 ~0.4）**。
- **Phase 2（完全态，probable next step）**：用 **YOLOv8-seg 在 35 GT 多边形+增强上微调**（或 Phase 1 检测器作 teacher 伪标 ~100-200 帧后微调）替入。触发条件 = Phase 1 实测 mask-IoU 确认 ≈0.4（即 train-free 路线触顶）。

### 数据需求（明确）
- Phase 1：**0 新标注**（只用 35 帧评测）。
- Phase 2 若触发：建议标注预算 **+65~165 帧（总量 ~100-200）** 跨更多视频，避免过拟合 9 场景；或依赖预训练+YOLOv8-seg 在 35 帧上弱监督微调（需验证收敛）。
- 弱监督可行性：✅ 用 Phase 1 检测器伪标大量无标帧 → 扩训练集 → 微调 seg。

## 3. ★ Q2 静止利用设计（时间聚合检测，非首帧检测）

> ⚠️ **改定 2 — 首帧检测太脆，改时间聚合（cc 复核要求）**
> 原「首帧跑重检测存 ref_poly」的问题：红灯下车**早停在斑马线上**，首帧很可能**本身就被车挡**，无干净帧可存。
> 更稳：**跨帧时间聚合条纹响应**（整段 median / 在每像素取多帧 max）→ 移动的车会被平均/取大时抵消，静止的斑马条纹始终在 → **看穿「某些帧被遮挡」、恢复更完整的斑马线**。

斑马线静止（仅相机漂移）→ 不必逐帧检测。设计成**有状态检测器**（Q2 逻辑封装在检测器内，DAG 不动）：

- `CrosswalkDetectorV2.__init__`：内部 `state=None`，维护 `accum`（条纹响应累积图，与帧同尺寸）。
- `detect(frame, vehicle_boxes)`：
  - **累积阶段（state=None / 未满 N 帧）**：对当前帧算**条纹密度响应图**（行/列方向梯度 + 亮暗周期），与 `accum` 做 **时间 median / per-pixel max** 累积；累积满 N 帧（或视频尾部）后 → 由聚合图做一次**四边形拟合**（见 §B）存 `ref_poly` + 锚点；`state` 置位。
  - **后续帧（已置位）**：估计帧间**全局漂移**（crosswalk ROI 内光流/特征点平移）→ 将 `ref_poly` warp 到当前帧 → 返回掩膜。漂移估计不可靠时回退到聚合图重拟合。
- **残余局限（必须写明）**：若**违章车全程一动不动压在远端斑马线上**，时间聚合也救不了那块被恒遮区域 → 只能靠 **Phase 2 seg 的学习先验**（"脑补"遮挡延续）补回。这是 train-free 路线的已知上限，已在改定 1 量化（~0.4）。
- **Phase 1a（最简）**：聚合满后**整片保持固定**多边形（忽略漂移）。mask-IoU 宽松（预期 ~0.4），直接根除 flicker（不再逐帧 argmax 跳变）。
- **Phase 1b**：加漂移矫正（warp），对齐 GT 逐帧多边形（GT 评测是 per-frame 漂移感知的）。
- DAG `n_crosswalk`（dag.py:79-84）**不改**：仍每 `cw_int` 调 `detect`，重活只在聚合满/rare 触发。检测器每视频新建 → 无跨视频污染。

> 收益：① flicker 根因从源头消除（11 漏检有望回）② 检测算力从「每 4 帧」降到「每视频 ~1 次聚合」③ **时间聚合天然抗「部分帧遮挡」**，比首帧检测鲁棒。

## 4. ★ Q3 denom=box 落地 + 阈值重标（扫描，非拍脑袋）

- **落地**：`occ_denom="box"` 已支持。语义 =「车足迹压线比例」（geometry.py:23-53, denom=box: inside/box_area）。
- **阈值解耦**：当前 `BatchViolationEngine.overlap_thr = p["overlap"]` 对两种 denom 共用（violation_engine.py:143）。改为：denom=box 时用新增的 `p["box_overlap"]`，mask 仍用 `p["overlap"]`（保留 D2 现状，向后兼容）。
- **重标方案（必做，扫描）**：
  1. 用新检测器（Phase 1a/b）+ `occ_denom="box"` 跑 9 正例 + 2 负例（01/10）。
  2. 在候选 `{0.10, 0.15, 0.20, 0.25, 0.30}` 扫 `box_overlap`。
  3. 选**最大化 F1 且保持：负例 01/10 FP=0、7 好视频 TP 不降、FP 不增**的阈值。
  4. 报告阈值-F1 曲线（含各视频 TP/FP），交 cc 复核。
  - 先验：天花板用 0.20+box→0.941，但那是**完美 GT mask**；真实检测器 overlap 分布会偏移，必须重扫，结果可能≠0.20。

## 5. ★ Q4 接地点/footprint

- 天花板用 `footprint=0.5`（车下半框）已达 0.94 → **保留默认 footprint=0.5**（已证 L0/L1 够）。
- 可选精修（非必须）：暴露 `footprint_mode ∈ {box_lower_half(默认), contact_point(L1: 框底边接地线)}`，contact_point 仅当 Phase 1 后仍有边际收益时做。3D/单应矫正(L2) 本次不做。

## 6. 红线执行（cc 复核重点①）

- **GT 绝不进生产**：新检测器 `detect(frame)` 只吃帧，不得 import/read `datasets/gt/**`、`crosswalk/*.json`、不得用 `GtCrosswalkDetector` 作默认。
- **无 per-video 硬编码多边形**：`config.yaml` 生产路径不写任何视频专属斑马线坐标（那是 light_priors 式过拟合，本项目痛点）。
- **验证手段**（写码时一并落地）：
  - 单测 `test_no_gt_leakage.py`：断言生产检测器模块源码不含 `datasets/gt`、`crosswalk/*.json`、`GtCrosswalkDetector` 引用。
  - `GtCrosswalkDetector` 仅留 `diag_gt_crosswalk_ceiling.py`（诊断）；默认 `comp["crosswalk"]` 永远是新帧推断检测器（cli.py:75）。
  - 评测脚本读 GT 属「评测/训练」用途，合法；红线只约束**生产推断**。
- **不回退 7 好视频**：全量 `eval_violations.py`（禁 --reuse、含负例）逐视频核 TP/FP vs 基线。
- **mask-IoU bar**：`eval_crosswalk_mask.py` 目标均值 **≥0.5**（现状 0.011）；任何视频 <0.5 标红。
- **04 出局**：fn 可接受，不计入成功标准。

## 7. 验证口径

- 模块：`eval_crosswalk_mask.py` mask-IoU 均值 **≥0.5**（vs 0.011）；逐视频，<0.5 告警。
- 端到端：`eval_violations.py`（11 视频，禁 --reuse）→ F1 **逼近 0.94**；**7 好视频(02/03/05/06/07/08/09) TP 不降、FP 不增**；**11 回（原 fn）**；**负例 01/10 FP=0**；04 fn（出局）。
- 消融：同检测器报 denom=mask vs box，确认 box 更优（如天花板所示）。

## 8. 分期

- **Phase 1（本次计划范围，train-free 探针）**：改进CV出四边形掩膜 + Q2 时间聚合锚定 + Q3 denom=box+阈值重扫。零新标注。**预期 mask-IoU~0.4（车底遮挡，不达 bar），仅验证通路+根除 flicker，不是终结解。**
- **Phase 2（probable next step）**：YOLOv8-seg 微调替入（35 帧+增强，或伪标扩量）。触发 = Phase 1 实测确认 train-free 触顶(~0.4)。需标注预算 + 算力。

## 9. 待 Jacob 拍板点

1. **denom=box 是否正式推翻 D2**（采用车百分比）？计划推荐「是」，并解耦 `box_overlap` 阈值（不影响 mask 路径）。
2. **过渡态先上 vs 直接 seg**？计划推荐过渡态（train-free）先上。
3. **若走 Phase 2 需标注预算**？计划估 +~100-200 帧；但 Phase 1 不需。

## 10. 文件清单 / 任务分解（TDD 先行）

- 新增：`src/redlight/models/crosswalk_v2.py`（有状态四边形检测器：条纹剖面+四边形拟合+**时间聚合锚定**+漂移 warp）+ 单测 `tests/unit/test_crosswalk_v2.py`（合成条纹图验证四边形拟合、时间聚合抗遮挡、无 GT 引用）。
- 改：`src/redlight/pipeline/tracker.py` 的 `SENSITIVITY_PRESETS` 加 `box_overlap`；`violation_engine.py` 按 `occ_denom` 取对应阈值。
- 复用：`cli.run(crosswalk_detector=, occ_denom=)` 注入验证；`eval_crosswalk_mask.py` / `eval_violations.py` 作回归。
- 阈值扫描脚本：在 `scripts/` 加 `sweep_box_overlap.py`（9 正+2 负，出曲线）。
- 红线单测：`tests/unit/test_no_gt_leakage.py`。

## 11. 第一步（计划获批后）

1. TDD 写 `crosswalk_v2.py` 四边形拟合（合成条纹图）。
2. 接 `cli.run(crosswalk_detector=CrosswalkDetectorV2(cfg), occ_denom="box")` 跑单视频看 mask-IoU。
3. 阈值扫描定 `box_overlap`。
4. 全量 eval 核 7 好视频不回退 + 负例 0 误报。
