# 度量修复完成报告（Plan v4 · eval-first）

- **作者**：wb（写码执行）
- **日期**：2026-07-16
- **计划**：`docs/plans/2026-07-16-wb-plan-v4-fix-eval-measurement.md`
- **cc brief**：`docs/handoff/2026-07-16-cc-direction-fix-eval-measurement.md`
- **状态**：Part A 收官（含全 11 视频权威数字）；Part B 代码就绪，等 Jacob 标注
- **红线**：头条 1:1 未动 · 只改评测工具不碰管线 · 负例读 `videos.csv` · Part B 不做全量 GT · 碎片仅度量分离不改判定

---

## 一、一句话结论

评测的"尺子"已修好：**头条口径一字未改**，同时新增两把辅助尺——把原来被隐藏的负例纳入、把 7 个 FP 拆成"真误报 vs 碎片"。全 11 视频跑通，cc 点名的 3 个验证锚点全部对上。

---

## 二、Part A — 端到端度量修复（已完成）

### 2.1 改了什么

| 文件 | 改动 | 是否碰判定 |
|---|---|---|
| `src/redlight/evaluation/violation_eval.py` | 新增 `load_video_metadata()`、`classify_false_positives()`；`aggregate()` 增 `true_fp_total/fragment_total/p_only_true_fp` | 否。`match_violation_events` **一字未动** |
| `scripts/eval_violations.py` | 默认视频集扩到 11（读 `videos.csv`）；逐视频分类 + 报真误报/碎片明细 + 诊断行；`_read_violations_csv` 补读灯态 | 否（评测脚本） |
| `src/redlight/app/cli.py` | `run()` 加 `return_track_samples=False`（为 Part B 取数）| 否。纯加法、默认 False、仅返回已有 `_track_samples`、零行为改动 |

### 2.2 全 11 视频权威结果（fresh 流水线）

产物：`data/output/eval_violations/report_full_2026-07-16.txt`

**头条（1:1，口径未变）**
```
P=0.500  R=0.778  F1=0.609   (tp=7  fp=7  fn=2)
命中窗平均覆盖率=0.426   车牌命中=2/7
```

**新拆解（A2）**
```
7 个 FP = 真误报 3（负例+窗外）+ 碎片 4（GT 窗内过分割）
诊断 P(仅真误报) = TP/(TP+真误报) = 7/(7+3) = 0.700
```

### 2.3 逐视频明细

| 视频 | 类型 | tp/fp/fn | 真误报 | 碎片 | 说明 |
|---|---|---|---|---|---|
| 违章01 | 负例 | 0/1/0 | **1** [48.4-60.6]绿灯 | 0 | ⭐ 之前被 eval 隐藏，现显式现身 |
| 违章02 | 正例 | 1/1/0 | 0 | **1** [32.7-34.1] | 命中 GT[21-68]，碎片重叠同窗 |
| 违章03 | 正例 | 1/0/0 | 0 | 0 | 完美 |
| 违章04 | 正例 | 0/0/**1** | 0 | 0 | ⚠️ 漏检 [42-43]（flicker 根因，v3 挂起） |
| 违章05 | 正例 | 1/2/0 | 0 | **2** [3.1-5.0][26.3-48.2] | 命中 + 两碎片同窗 |
| 违章06 | 正例 | 1/0/0 | 0 | 0 | 完美 |
| 违章07 | 正例 | 1/2/0 | **1** [44.2-49.4]窗外 | **1** [4.4-12.7] | 车牌命中 1/1 |
| 违章08 | 正例 | 1/0/0 | 0 | 0 | 完美，车牌命中 1/1 |
| 违章09 | 正例 | 1/1/0 | **1** [94.1-95.8]窗外 | 0 | ⭐ 窗外真误报（与 GT[11-72] 零重叠） |
| 违章10 | 负例 | 0/0/0 | 0 | 0 | ✅ 负例零误报（干净） |
| 违章11 | 正例 | 0/0/**1** | 0 | 0 | ⚠️ 漏检 [15-28]（flicker 根因，v3 挂起） |

### 2.4 cc 点名验证锚点（全部通过）

- ✅ **01 显式现身为负例真误报** —— 之前 eval 只跑 9 正例，01 根本不进评测，这次它带着 `[48.4-60.6]绿灯` 的误报被抓出。
- ✅ **09 窗外真误报** —— `[94.1-95.8]` 与 GT[11-72] 零重叠，判"窗外真误报"正确。
- ✅ **02 碎片** —— `[32.7-34.1]` 与 GT[21-68] 有重叠但 1:1 没配上，判"碎片"正确。
- ✅ **`return_track_samples` 零行为改动** —— 新分支在 `decide()` 全部完成之后、仅当形参为真才触发；默认 False 时返回值与原来完全一致。

### 2.5 度量修复带来的洞察

原头条 **P=0.50 看着很差**，但拆开看：7 个"错误"里有 **4 个只是同一违章被切成多段（碎片）**，并非冤枉好人；真正的"误抓" 只有 3 个。**产品真正在乎的"冤枉好人率" P(仅真误报)=0.70**，比头条乐观得多。这正是 cc 要的"度量分离看清楚"——**头条诚实保留，诊断行揭示真相**。

> 注：头条 F1 从记忆里的旧基线 0.636 变为本次 0.609，P/R（0.500/0.778）不变，差异来自本次全 11 视频 fresh 重跑（口径与算法均未改）。以本次为准。

---

## 三、Part B — 模块级评测（代码就绪，等标注）

### 3.1 已交付代码（均可单测，不碰管线）

| 文件 | 作用 |
|---|---|
| `src/redlight/evaluation/module_metrics.py` | 纯函数：`band_iou` / `mask_band` / `iou_box` / `attribution_union` / `stationary_accuracy` |
| `scripts/gen_gt_skeleton.py` | 从 `events.csv` 生成 GT 骨架（时间戳/窗口预填，像素坐标留空） |
| `scripts/eval_crosswalk_mask.py` | **B1**：斑马线掩膜 band-IoU（`detect(frame)` 不带 vehicle_boxes，主指标） |
| `scripts/eval_tracking.py` | **B2**：`cli.run(return_track_samples=True)` → ID 碎片化 + 静止判定准确率 |

### 3.2 B2 归属规则（cc 收紧后写死）

碎片化计数 = **窗内任一帧中，框与锚框 IoU ≥ T(=0.5) 的所有 `track_id` 之并集大小**（理想=1），**非**只取 IoU 最高那个——否则会低估碎片化。已在代码与计划中固化。

### 3.3 等 Jacob 标注（wb 不代标）

- **B1**：`datasets/gt/crosswalk/违章NN.json`（9 正例），每视频 ~4 帧，填 `y0/y1`（斜带按关键帧独立标以兼容相机漂移）。合计约 **36 个 y-band**。
- **B2**：`datasets/gt/tracking/违章NN.json`（9 正例），每窗 1-2 锚帧，填 `ts + box`。合计约 **9-18 个框**。
- 标完直接跑 `eval_crosswalk_mask.py` / `eval_tracking.py` 出首个模块级基线。**其中 B1 会把 Step 0 手工发现的"11 候选缺失 62.9%"变成可回归指标。**

---

## 四、测试与回归

```
tests/unit/  228 例全绿
  ├─ test_violation_eval.py     4 例（Part A：负例/碎片/窗外/聚合）
  └─ test_module_metrics.py     9 例（Part B：band-IoU / 并集归属 / 静止准确率）
无回归。
```

---

## 五、v3（管线修 04/11）状态

**正式挂起**。本次度量已量化：04/11 漏检 = fn=2，是当前仅有的两个漏检，均为斑马线条带 flicker 根因。等 Part B 掩膜 band-IoU 基线出来、看清掩膜失准幅度后再决定是否重启 v3。

---

## 六、待办 / 交接

1. **cc 验收 Part A**：真误报/碎片拆解数已对上预期锚点（01 现身、09[94-96]窗外、02[32-34]碎片、`return_track_samples` 零行为改动）。
2. **Jacob 标注 Part B**：填 `datasets/gt/crosswalk/*.json` 的 `y0/y1` 与 `datasets/gt/tracking/*.json` 的 `box`。
3. **commit**：本次改动尚未提交（等 cc 验收放行）。改动清单见 `git status`（4 改 + 11 新增）。
