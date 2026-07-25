# 方法：量「误绿一次」(Fork A 核心产出) — cc review 口径

> 出自 wb(executor)。cc brief: `docs/handoff/2026-07-26-cc-forkA-scouting-and-measure-brief.md` §3。
> 本文**只出方法，不跑**。cc 审过口径后再执行。红线不变：只测量不接线、生产 prior/权重不动、禁 GT 进推理、去循环口径。

## 1. 目标与红线
- **目标**：把「当前管线误绿多严重」从猜变数字 —— 决定「A 收工 vs 转 B 建检测器」的标尺。
- **红线(逐条贴 brief §5)**：
  - 只测不接线；`dag.py` / `enforce_transition_limit` / 生产路径**不碰**。
  - 生产 `configs/light_priors.json` 与 `models/ped_signal.pt` **只读不写**。
  - **禁 GT 进推理**：选灯走 GT-free 路径(L1/L2/L3)，GT 只做比对/评分。
  - **去循环**：禁用「在 prior 处采样再比同一 GT」的自证指标(Diag2 坑)。
  - scoped git、trunk main、TDD 先、公开数据许可逐个核(本次不下载任何外部数据)。

## 2. 定义
- **误绿(帧级)**：一帧里管线输出 `walk`(绿)，但真值 `gt_walk=False`(即该帧真值灯态≠green：red/occluded/none)→ 可能触发假违章。
- **可评帧(evaluable frame)**：该帧存在 ≥1 候选(YOLO∪HSV 检出)，即管线「有可能输出绿」。无候选=管线输出空，不计误绿也不计漏绿(漏绿=假负绿，不在本次范围，仅作附注)。
- **误绿事件(事件级)**：连续若干可评帧都 `walk & ¬gt_walk` 的-run = 1 个误绿事件(=一次可能误报违章)。
- **事件级误绿率** = 误绿事件数 / 管线总绿事件数(总绿事件=所有 `walk` 连续 run，不论 GT)。即「管线报绿里有多少是假的」。

## 3. 输入资产(已盘点，路径属实)
| 资产 | 路径 | 用途 | 是否 GT |
|---|---|---|---|
| 真值灯态(事件级) | `datasets/gt/events.csv`(11 视频全覆盖, light_state+is_violation) | 帧→段→gt_walk 映射 | **是(GT,仅比对)** |
| 真值灯框 | `datasets/light_location_gt.json` `annotations[].true_box_norm` + `derived_per_video.true_center` | 成因归因(选错灯?) | **是(GT,仅比对)** |
| 生产分类器 | `models/ped_signal.pt`(config `models.ped_signal_model`) | 状态判别(walk/stand/off) | 否(现有模型) |
| 生产 prior | `configs/light_priors.json` | L1 高斯中心 + crop-misframe 对照 | 否(手写,非 GT) |
| 选灯 L1/L2/L3 | `src/redlight/models/ped_light_selector.py` `select_gtfree`(GT-free) | 当前最好选灯 | 否 |
| L3 判别头 | `src/redlight/models/l3_ped_vehicle.py` | 选灯 L3 评分 | 否(训于 Jacob 标注,非 GT) |
| 候选提取 | `scripts/collect_candidates.py` | 逐帧 YOLO∪HSV 候选 | 否(检测器) |
| 源视频 | `input_video/违章XX.mp4`(fps≈29.70) | fi→时间映射 | 否 |
| 标注 | `data/output/l3_labels/labels.json`(203 簇) + manifest | 训全量 L3 夹具 | 否(人工标注) |

> 注：`datasets/gt/light_state/*.csv` 的 `gt_state` 列**实际为空**，不用；权威帧级真值走 `events.csv` 段映射。

## 4. 前置步骤(执行期才做，本次仅列)
1. **密集候选提取**：重跑 `collect_candidates.py` 走**逐帧密集模式**(全部 11 视频，每 `step` 帧一采，记录**源帧号** source_fi)，产出 `{video: {source_fi: [candidates]}}`。→ 当前 `candidates_temporal.json` 仅 33 条(只 GT 标注帧)，**不足以做帧级测量**，必须重抽。YOLO 跑 11 视频约数千帧，属一次性成本。
2. **训全量 L3 夹具**：`l3_ped_vehicle.train_model` 在**全部 797 裁图**(manual 203 + auto_ped 48 + auto_other 546)上训一个单模型，作「当前最好选灯」权重。→ 评测用的是 per-video LOVO，生产/测量需全量单模。训于 Jacob 标注，**非 GT 播种**，满足去循环。
   - 备选(更保守去循环)：直接报 **L1+L2-only**(无 L3)作基线行；L1+L2+L3(全量)作主行。两行都报。

## 5. 测量流程(逐帧)
对每视频 V、每可评帧 f(source_fi → t=f/ fps)：
1. **候选**：取该帧 YOLO∪HSV 候选 `candidates`(含 box_norm/source)。
2. **选灯(GT-free)**：`compute_temporal_scores(V 全部帧)` → `temporal_scores`；`l3_scores` = 全量 L3 对每候选的 P(ped)；`select_gtfree(candidates, ped_prior, temporal_scores, l3_scores, l3_weight=best)` → `selected`(box_norm)。**无任何 GT 参与。**
3. **裁图+判别(现有分类器)**：
   - `crop_sel` = 按 `selected.box_norm` 裁帧；`out_sel = classifier.classify(crop_sel)` → (label_sel, conf_sel)。
   - `crop_prior` = 按 `light_priors.json` 的 prior ROI 裁帧(复现生产 crop 来源)；`out_prior = classifier.classify(crop_prior)`。
4. **真值比对(仅读)**：`t` → `events.csv` 命中段 → `gt_walk = (段.light_state=='green')`；`true_box` = `light_location_gt` 该视频真值框。
5. **记录**：`{video, fi, t, out_sel, conf_sel, out_prior, selected_box, prior_box, true_box, gt_walk}`。

## 6. 指标
- **帧级误绿率** = (#帧 out_sel=='walk' & ¬gt_walk) / (#可评帧)。
- **事件级误绿率** = (#误绿事件) / (#管线总绿事件)。
- **11 视频分解**：每视频上述两率 + 可评帧数 + 绿事件数。
- **成因归类**(对每条误绿帧，GT 仅比对)：
  - **车灯误选**：`selected_box` 与 `true_box` IoU<0.3 或中心距>0.04 → 选灯错(L3/selector 责任)。
  - **crop 偏框**：`selected_box≈true_box` 但 `out_prior=='walk'` 且 `out_sel≠'walk'` → 生产 prior-ROI 裁图错位(偏框毒化病灶)。
  - **状态判错**：`selected_box≈true_box` 且 `out_sel=='walk'` & `¬gt_walk` → 分类器把对齐的真行人灯判绿(暗绿/反射误判)。
  - 各成因占比(占误绿帧)。
- **附注**：漏绿率(真值绿但管线未输出绿)作次要指标，不计入主误绿率。

## 7. 去循环保证(逐条对应 brief 硬要求)
- 选灯全程 `select_gtfree` GT-free(护栏1)：无 `gt` 参数；`ped_prior` 取自 `light_priors.json`(手写,非 GT)；L3 训于 Jacob 标注(非 GT 播种)。
- GT(`events.csv` / `light_location_gt.json`)**只进第 4–5 步比对**，绝不进 1–3 步推理。
- 不重训生产分类器、不采 prior 自比、不碰 `dag.py` / `enforce_transition_limit`。
- 双行报告(L1+L2-only 基线 + L1+L2+L3 主行)暴露 L3 自身的去循环性。

## 8. 交付
- `docs/reports/2026-07-26-wb-falsegreen-measure.md`：总体/事件级误绿率 + 11 视频分解 + 成因占比 + 双行(L1+L2 vs L1+L2+L3)。
- 误绿样本画廊(HTML)：每条误绿帧叠画 GT 红框(true_box) + 选中绿框 + prior 虚框 + crop_sel/crop_prior 缩略 + out_sel/out_prior + GT 灯态，供 Jacob 抽检。

## 9. 需 cc 拍板的点(审口径时定)
1. **状态判别源**：主行用 `ped_signal.pt`(分类器路径)？还是同时报 `method=color`(生产默认颜色兜底)对照？→ 建议主行分类器 + 附 color 对照。
2. **L3 权重**：取 LOVO 最优 `w=1.0`(纯 L3)还是稳健 `w=0.5~0.7`？→ 建议主行 `w=0.7`、附 `w=1.0` 对照。
3. **密集候选 step**：建议 `step=4`(与现有候选集一致、降成本)；是否要 `step=1`(逐帧,更密但 YOLO 成本高)？
4. **漏绿**是否纳入报告(次要，不计入主率)？
5. **fps 映射**：源视频 fps≈29.70，密集提取记录 source_fi，`t=source_fi/fps` 映射 `events.csv`；段间空隙帧按 `gt_walk=False`(保守)处理，是否同意？
