# B1 方案：斑马线 v2 时序聚合激活 + denom=box + 阈值重扫

日期：2026-08-03 | 作者：qw | 状态：交 cc 审

承 Phase A 诊断（`docs/reports/2026-08-03-qw-crosswalk-recall-ceiling.md`, commit `237b49d`）+ cc 两层复核通过（`docs/handoff/2026-08-03-cc-verify-qw-crosswalk-recall-ceiling.md`）+ v6 计划（`docs/history/plans/2026-07-17-wb-plan-v6-crosswalk-detector-upgrade.md`）。**接着 v6 计划 + 现有 `crosswalk_v2.py` 扩展，不重造。**

---

## 1. 诊断指向（本方案的事实依据）

Phase A 诊断数据（35 帧 × 9 视频，cc 两层 bit-for-bit 复核通过）：

| 指标 | v11（生产默认） | v2（单帧独立实例） |
|---|---|---|
| mask-IoU 均值 | **0.011** | **0.249** |
| recall 均值 | 0.012 | — |
| precision 均值 | 0.093 | — |
| IoU<0.5 的帧 | 35/35 | 多数 |

关键发现：
- v11 根因不是"全宽带太宽溢出"（cc spec 原预期，已被 qw 数据推翻），而是**薄带定位错高度 + 出不了透视梯形**——有效覆盖近零。
- v2 梯形方向可行：05@3.0s v2 recall=0.917，梯形确实抓到了透视斑马线。
- v2 单帧 IoU=0.249 仍多数 <0.5，但这是**每帧独立实例**的结果——running-max 时序聚合**根本没跑**。

**核心问题**：v2 的 running-max 时序聚合代码已存在于 `crosswalk_v2.py:73-80`（`self._accum = np.maximum(self._accum, resp)`），但 `eval_crosswalk_mask.py:52` 每帧新建独立实例 → `_accum` 每帧重置 → 聚合形同虚设。生产路径 `cli.run` 每视频新建一次检测器（`cli.py:75`），时序聚合**天然能跑**——但从未被量过。

**优化方向**：让 v2 的 running-max 时序聚合真正跑起来（跨帧累积条纹响应，移动车被 max 掉、静止斑马条纹留下 → 看穿部分帧遮挡），把 mask-IoU 从 0.249 往 v6 预期的 ~0.4 抬。

---

## 2. 具体改法

### 2.1 评测侧：让 v2 走完整视频时序聚合（核心改动）

**改 `scripts/eval_crosswalk_mask.py`**：
- 现状：每个 GT anchor 帧新建独立 `det` 实例（`:52`），v2 的 `_accum` 每帧重置。
- 改法：v2 模式下，**每视频建一个实例**，先顺序跑完整视频（按 `cfg.inference.fps` 采样），在跑到每个 GT anchor 帧的 ts 时记录当前 mask，与 GT 对比。
- 理由：running-max 是顺序无关的（`np.maximum` 交换律），但需要**足够帧数**累积才有意义（4 帧 GT 远不够，生产是 ~200 帧/视频）。
- 实现要点：
  - 复用 `VideoSampler` 或 `gd._read_frames_at` 顺序读帧
  - v2 实例在视频开始时 `__init__`，逐帧调 `detect(frame)` 更新 `_accum`
  - 到 GT anchor ts 时，记录当前 `det._accum` 的 `_fit_trapezoid` 结果
  - v11 无状态，保持每帧独立实例（口径不变）
- **不改生产代码**：只改评测脚本。

### 2.2 检测器侧：crosswalk_v2.py 参数微调（可选，看 2.1 结果）

v2 的 `_fit_trapezoid` 和响应图参数有几处可能需要调：
- `bright_thr = max(160.0, mean_v + 0.8*std_v)`（`:61`）：部分暗视频（03/09 的 V 均值低）可能阈值过高 → 可降 160→140 或调系数 0.8→0.6
- `band_thr = 0.10 * maxc`（`:95`）：梯形两端变窄时可能丢条纹 → 可调 0.10→0.05
- `roi` 从 `0.35*h` 开始（`:69`）：部分视频斑马线位置偏高 → 可调 0.35→0.30
- **不急着改**：先跑 2.1（时序聚合激活）看 mask-IoU 升到多少，如果到 ~0.4 就够（v6 预期），参数微调留作迭代。

### 2.3 生产注入：cli.run(crosswalk_detector=, occ_denom="box")

**已存在，不改注入机制**（v6 计划 §1 已确认）：
- `cli.run(crosswalk_detector=CrosswalkDetectorV2(cfg), occ_denom="box")` 直接可用
- DAG 的 `n_crosswalk` 节点（`dag.py:79-84`）每 `cw_int` 帧调 `det.detect(frame)`，v2 天然跨帧累积
- 检测器每视频新建（`cli.py:75`）→ 无跨视频污染
- **唯一要改的**：让 `BatchViolationEngine` 在 `occ_denom="box"` 时用 `box_overlap` 阈值（见 2.4）

### 2.4 box_overlap 阈值解耦 + 重扫

**改 `tracker.py` 的 `SENSITIVITY_PRESETS`**：
- 现状：只有 `overlap`（给 denom=mask 用），balanced=0.20
- 改法：每档 preset 加 `box_overlap`（初值 0.20，待扫描后定）
- 兼容：`box_overlap` 缺失时回退到 `overlap`（不破坏旧口径）

**改 `violation_engine.py`**：
- 现状：`overlap_thr = p["overlap"]`（`:143-144`），不分 denom
- 改法：`overlap_thr = p.get("box_overlap", p["overlap"]) if occ_denom == "box" else p["overlap"]`

**新 `scripts/sweep_box_overlap.py`**：
- 9 正例 + 2 负例（01/10），用 v2 + `occ_denom="box"`
- 扫 `box_overlap ∈ {0.10, 0.15, 0.20, 0.25, 0.30}`
- 选最大化 F1 且满足：负例 01/10 FP=0、7 好视频 TP 不降、FP 不增
- 报阈值-F1 曲线交 cc 复核

### 2.5 红线单测

**新 `tests/unit/test_no_gt_leakage.py`**（v6 计划 §6 已设计）：
- 断言 `crosswalk_v2.py` 源码不含 `datasets/gt`、`crosswalk/*.json`、`GtCrosswalkDetector` 引用
- 已有现成参考：当前 `tests/unit/test_no_gt_leakage.py` 已检查 `crosswalk_v2.py` 和 `cli.py`，扩展即可

---

## 3. 怎么验

### 3.1 近端：mask-IoU before/after

用改后的 `eval_crosswalk_mask.py`（v2 走完整视频时序聚合）：
- **Before**：v2 单帧独立实例 = IoU 0.249（Phase A 已量）
- **After**：v2 完整视频时序聚合 → 预期升到 ~0.3-0.4（v6 改定 1 预期）
- **Gate**：mask-IoU 有升（必要条件），不硬卡 ≥0.5（train-free 大概率触顶 ~0.4，v6 已诚实下调）

### 3.2 北极星：端到端违章 F1

两条口径并行跑：
- **`eval_violations.py`**（11 视频，禁 `--reuse`）：v2 + `occ_denom="box"` + 扫描后的 `box_overlap` 阈值
  - 对比基线 v11 + denom=mask：F1 0.636（v6 计划数据）
  - 目标：逼近 0.94（GT 天花板），但 train-free 预期触顶 ~0.8
- **`diag_gt_crosswalk_ceiling.py`**（喂真 detector 而非 GT mask）：
  - 对比"GT mask 注入"（0.941）vs "v2 真检测器"——量 v2 离天花板还差多少

### 3.3 安全检查

- **负例 01/10**：v2 检测的 mask 在无违章视频上是否产生新 FP → 必须 0 新误报
- **7 好视频**（02/03/05/06/07/08/09）：v2 的 TP 不降、FP 不增
- **flicker**：v2 时序聚合后 mask 是否稳定（不再逐帧跳变）——定性检查
- **04/11**：fn 可接受（灯态/零检出局，不计入成功标准）

---

## 4. 风险

| 风险 | 机制 | 对策 |
|---|---|---|
| 负例 01/10 新误报 | v2 的梯形 + running-max 可能检测到非斑马线条纹（人行道/车道线/建筑阴影） | 扫描时检查 01/10 FP=0；若有，调 bright_thr/band_thr 或加面积下限 |
| 7 好视频回退 | v2 mask 形状/面积与 v11 不同 → overlap 分布偏移 → 原阈值不适用 | 阈值重扫（2.4）；先跑单视频看分布再铺全量 |
| 前 N 帧积累不足 | 生产 DAG 前 `cw_int` 帧时 `_accum` 只有 1-4 帧 → mask 不完整 | 可接受（违章需持续 `duration` 帧才触发，前几帧不够 duration）；或预热（先跑 N 帧再开始判定） |
| 手持机位漂移 | running-max 在相机大幅移动时会"糊"——不同位置的条纹叠加 | v6 计划 Phase 1b 设计了漂移矫正（warp），但 Phase 1a 先不做（看效果再说） |
| 触顶 ~0.4 | v6 改定 1 已诚实预期：train-free 看不见车底 60% 遮挡 | 如确认触顶，记 Phase 2（seg 微调）为独立立项交 Jacob，不硬凑 |

---

## 5. 执行顺序（cc 审通过后）

1. **改 `eval_crosswalk_mask.py`**：v2 走完整视频时序聚合 → 量 mask-IoU before/after
2. **改 `tracker.py` + `violation_engine.py`**：加 `box_overlap` + denom 分流
3. **新 `sweep_box_overlap.py`**：9 正 + 2 负，扫阈值定 `box_overlap`
4. **全量 `eval_violations.py`**：v2 + box + 新阈值 → 端到端 F1
5. **红线单测**：`test_no_gt_leakage.py` 扩展
6. **交付 cc 独立验效果**

所有改动在独立 worktree 里做，scoped add，原子写（tmp+os.replace），署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`。

---

## 6. 红线（继承 v6 §6）

- **GT 绝不进生产**：新检测器 `detect(frame)` 只吃帧，不 import/read `datasets/gt/**`、`crosswalk/*.json`、不用 `GtCrosswalkDetector` 作默认
- **无 per-video 硬编码多边形**：`config.yaml` 生产路径不写视频专属斑马线坐标
- **不静默替换默认 v11**：新检测器走注入/flag，`cli.run` 默认仍是 `CrosswalkDetector`（v11）；cc 验完 Jacob 拍板才接线
- **不回退 7 好视频**：全量 `eval_violations.py`（禁 `--reuse`、含负例）逐视频核 TP/FP
- **scoped add**：只提交自己改的文件，绝不 `git add -A`

---

**一句话**：v2 的 running-max 时序聚合代码已存在但被 eval 每帧新建实例架空了。核心改动 = 改 eval 让 v2 走完整视频时序聚合 + 加 box_overlap 阈值解耦 + 扫描定阈值 + 端到端验 F1。预期 mask-IoU 从 0.249→~0.4（train-free 触顶），不硬卡 0.5；负例零新误报 + 7 好视频不回退 + flicker 根除才是真 gate。
