# 方法：量「误绿一次」(Fork A 核心产出) — cc review 修正版(已放行)

> 出自 wb(executor)。基于 cc review `docs/handoff/2026-07-26-cc-review-falsegreen-method-conditional.md`。
> 去循环/GT只比对/双行/成因归类/交付形态 **已过审**;本版仅修 cc 挖出的 **A/B/C 三缺陷 + §5 两纪律 + 5 点拍板**。
> **方法主体已过,本版按 cc 要求补缺陷即可,现已可跑。**

## 1. 目标与红线
- **目标**：把「当前管线误绿多严重」从猜变数字 —— 决定「A 收工 vs 转 B 建检测器」的标尺。
- **红线(逐条贴 cc review §6)**：只测不接线；生产 `configs/light_priors.json` 与 `models/ped_signal.pt` **只读不写**；**禁 GT 进推理**；去循环口径。
- **两纪律(写进代码, cc review §5)**：
  1. **禁调 `seed_with_gt` / `derive_ped_priors`**(GT 播种/GT 派生 prior)。`ped_prior` 只用 `light_priors.json` 的生产位置 + **内联**硬编码几何先验(空列表默认值,不调用 derive_ped_priors)。
  2. **`light_state` 语义自查**：跑前拿 08(全绿)/01(全红)对一眼灯态方向,确认 `events.csv.light_state` 是**行人灯**态、没搞反,避免整份误绿反号。

## 2. 定义(含缺陷 A 三态映射)
- **误绿(帧级)**：管线输出 `walk`(绿)，但该帧真值 `gt_walk=False`(灯态=red/occluded/none)→ 可能触发假违章。
- **三态 GT 映射(缺陷 A 硬要求)**：`events.csv` 段 `.light_state` →
  - `green` → `gt_walk=True`(真绿)。
  - `red` / `occluded` / `none` → `gt_walk=False`(**计误绿**)。
  - `unknown` 段 + **段间空隙帧** → `gt_walk=UNKNOWN` → **移出可评帧**(既不计误绿也不计分母)。报告单列"因 unknown/空隙排除的帧数",透明化(标尺不灌水)。
- **可评帧**：有 ≥1 候选 且 `gt_walk≠UNKNOWN` 的帧。
- **误绿事件(事件级)**：连续可评帧都 `walk & ¬gt_walk` 的 run = 1 误绿事件。事件级误绿率 = 误绿事件 / 管线总绿事件。

## 3. 输入资产(已独立核验属实, cc review §0)
- 真值灯态(事件级)：`datasets/gt/events.csv`(11 视频全覆盖, light_state+is_violation) — **权威帧级真值**。
- 真值灯框/中心：`datasets/light_location_gt.json` `annotations[].true_box_norm` + `derived_per_video.true_center`(成因归因用; `light_state/*.csv` 的 `gt_state` 列空,不用)。
- 生产分类器：`models/ped_signal.pt`(config `models.ped_signal_model`) — **未接线、已知过拟合**,作对照。
- 生产 prior：`configs/light_priors.json` = `{video:[cx,cy,roi_px]}`(手写,非 GT) — L1 中心 + color 主行端到端裁图源。
- 选灯 L1/L2/L3：`src/redlight/models/ped_light_selector.py` `select_gtfree`(GT-free,无 gt 参数)。
- L3 判别头：`src/redlight/models/l3_ped_vehicle.py`(训于 Jacob 标注,非 GT)。
- 候选提取：`scripts/collect_candidates_dense.py`(**新建,纯 GT-free,从源视频**)。
- 源视频：`input_video/违章XX.mp4`(fps≈29.70) — **密集候选从源视频抽**(datasets/frames 仅稀疏图,不用于测量)。

## 4. 前置步骤(执行期做)
1. **密集候选重抽(纯 GT-free, 缺陷 A/B 要求)**：`collect_candidates_dense.py` 从 `input_video/*.mp4` 逐帧 `step=4`(≈7.4 采样 fps)抽 YOLO cls=9∪HSV 候选,记 **source_fi**(源帧号),**不写 gt_box_norm**,输出 `data/output/candidates_dense.json`(全 11 视频)。→ 现有 `candidates_temporal.json` 仅 33 条(含 GT 播种)不可用。
2. **训全量 L3 夹具**：`train_l3_full.py` 在**全部 797 裁图**(manual203+auto_ped48+auto_other546)训单模型 → `models/l3_ped_full.pt`。训于 Jacob 标注(非 GT 播种)。

## 5. 测量流程(逐帧, 含两纪律)
对每视频 V、每可评帧 f(source_fi → `t=source_fi/29.70`):
1. **候选**：取该帧 YOLO∪HSV 候选(box_norm/source)。
2. **选灯(GT-free)**：`compute_temporal_scores(V 全帧)` → temporal_scores；`l3_scores` = 全量 L3 对每候选 P(ped)；
   - no-L3 基线：`select_gtfree(cands, ped_prior, temporal_scores)`。
   - L3 w=0.5：`select_gtfree(cands, ped_prior, temporal_scores, l3_scores, l3_weight=0.5)`。
   - `ped_prior` = **内联** `{"aspect_mean":3.0,"aspect_std":1.0,"area_mean":0.005,"area_std":0.003}`(生产知识:行人灯竖长;**不调 derive_ped_priors**)。无任何 GT。
3. **状态判别(两路径)**：
   - **color 主行**(现役逻辑,复用 `TrafficLightDetector._color_state(roi)`)：
     - `out_prior = _color_state(crop_prior)` — `crop_prior` = 按 `light_priors.json` prior ROI 裁(=**真正现役端到端标尺**)。
     - `out_sel = _color_state(crop_sel)` — `crop_sel` = 按 selected box 裁(=best-location 变体)。
   - **classifier 对照**：`out_clf = SignalStateClassifier(ped_signal.pt).classify(crop_sel)` → walk/stand/off。
4. **真值比对(仅读)**：`t` → `events.csv` 命中段 → 三态映射(§2)得 `gt_walk`(UNKNOWN→本帧排除)。`true_center` 来自 `derived_per_video[V]`(成因用)。
5. **纪律②自检**(跑前一次性)：08 中段帧 crop_prior `_color_state` 应为 green；01 应为 red。不符 → 中止报错(灯态源搞反)。

## 6. 指标与报告行(含缺陷 B/C)
- **帧级误绿率** = (#误绿帧) / (#可评帧)；**事件级误绿率** = (#误绿事件)/(#总绿事件)。
- **11 视频分解** + **因 unknown/空隙排除帧数**(透明)。
- **漏绿附注**(拍板④)：真值绿但管线未输出绿(召回洞),不计主率。
- **报告 5 行**(选灯=best L1/L2/L3, 除 R0 端到端用 prior 位置)：
  | 行 | 状态 | 选灯 | 说明 |
  |---|---|---|---|
  | **R0 现役端到端** | color @ prior-ROI | prior(生产) | **真正现役标尺(cc B 字面)** |
  | **R1 color@best** | color @ selected | no-L3 | **主标尺(现役状态+最好选灯)** |
  | R2 color@best+L3 | color @ selected | L3 w=0.5 | 对照:L3 降误绿? |
  | R3 clf@best | classifier @ selected | no-L3 | 对照:分类器降误绿? |
  | R4 clf@best+L3 | classifier @ selected | L3 w=0.5 | 对照 |
  - 主标尺 = **R0(字面现役) + R1(修位置后)**;其余为 A 改进候选对照。
  - **弃 w=1.0**(sweep 噪声),只用 no-L3 + w=0.5(缺陷 C)。
- **成因归类**(对误绿帧,用 `true_center` 距离代理 true_box)：车灯误选(selected 离 true_center 远) / crop偏框(prior-ROI 判绿但 selected 裁图不判绿) / 状态判错(selected≈true_center 且 crop_sel 判绿但 GT 非绿)。各占比。

## 7. 去循环保证(逐条)
- 选灯全程 `select_gtfree` GT-free(无 gt 参数);L3 训于 Jacob 标注(非 GT 播种);`ped_prior` 内联(禁 derive_ped_priors/seed_with_gt)。
- GT(`events.csv`/`light_location_gt.json`)只进第 4 步比对,绝不进 1–3 步。
- 不重训生产分类器、不碰 `dag.py`/`enforce_transition_limit`、不接线。
- 双维度报告(R0/R1 + R3/R4)暴露 state/L3 各自的去循环性。

## 8. 交付
- `docs/reports/2026-07-26-wb-falsegreen-measure.md`：5 行误绿率 + 11 视频分解 + 排除帧数 + 成因占比 + 漏绿附注。
- 误绿样本画廊(HTML)：叠画 GT 红框(true_center 周边)/选中绿框/prior 虚框/crop 缩略/out 各路径/GT 灯态,供 Jacob 抽检。

## 9. 五点拍板落地(cc review §4)
1. 状态源 → **color 主行(R0/R1) + classifier 对照(R3/R4)**(缺陷 B)。
2. L3 权重 → **no-L3 基线 + 固定 w=0.5**,弃 w=1.0/0.7(缺陷 C)。
3. 密集 step=4 同意;重抽**纯 GT-free 不写 gt_box**(缺陷 A 配套)。
4. 漏绿纳入附注同意。
5. fps=29.70 映射同意;**空隙帧改判 UNKNOWN 排除**(缺陷 A)。
