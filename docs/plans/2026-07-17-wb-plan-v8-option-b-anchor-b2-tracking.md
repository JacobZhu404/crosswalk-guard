# Plan v8 — 选项 B：落袋 0.889，转移杠杆（B2 跟踪度量先行 · Phase 1.5 护栏暂缓待诊断）

> 出自：cc 方向 brief（Jacob 拍板 B）。灯态误绿手调门控已证死路（背心绿击穿紧凑度/面积），灯态整体延后立项。当前把 0.889 干净落袋，精力投到更便宜清晰的杠杆。
> 状态：revert 完成 + 锚点 0.889 已回填确认（见 §0）；B2 放行（GT-free 预览标"下界"、骨架不复生成）；Phase 1.5 暂缓（先诊断 05/09 根因）。本计划按 cc 复核精炼后待最终放行。

---

## 0. 已完成的无条件第一步（revert → 干净基线）

- `git checkout HEAD -- configs/config.yaml src/redlight/models/traffic_light.py` —— 还原失败门控（注意：brief 原 `git checkout <file>` 不带 HEAD 只从暂存区恢复=白还原；改动是 staged 状态，必须用 `HEAD`）。
- `rm tests/unit/test_traffic_light_validity.py` —— TDD 围绕失败门控写，一并撤。
- 3 份 docs 保留作"此路不通"历史：`docs/plans/2026-07-17-wb-plan-v7-light-false-green-fix.md`、`docs/reports/2026-07-17-wb-plan-v7-phase1-report.md`、`docs/reports/2026-07-17-wb-diag-gate-ineffective.md`（**不删**，下次灯态立项直接读避免重走）。
- 验证：`grep -c` 门控代码 = 0，工作树除 3 份 untracked docs 外与 HEAD(`f634885`) 一致，`git diff HEAD --stat` 为空。
- **锚点 eval（已跑完，✅ 确认回 0.889）**：`scripts/eval_violations.py --detector v2 --occ-denom box --preset balanced`（禁 `--reuse`，log=`/tmp/eval_anchor_0889.log`）→ **总体 P=0.889 R=0.889 F1=0.889 (tp=8 fp=1 fn=1)**，命中平均覆盖=0.904，车牌=5/7，拆解 真误报=1。基线干净落袋，作为任何新改动前的标尺。

> revert 后工作树 = HEAD，无需为 revert 本身提交。docs 保持 untracked 历史；若 cc 要求入仓，后续 scoped add + 署名提交。

---

## 1. 主任务：B2 跟踪模块评测（揭开最后一个盲区，只建度量不改代码）

### 1.1 现状与缺口
- 红绿灯(`eval_light_fast`)/斑马线(`eval_crosswalk_mask`)/占道(端到端覆盖)都有模块级 eval；**唯独车辆跟踪没有独立度量**。
- 跟踪断裂直接吃掉违章事件（覆盖率断续、episode 碎片），但现在完全看不见。
- 脚手架已存在（Plan v4 留的）：
  - `scripts/eval_tracking.py`：读 `datasets/gt/tracking/{video}.json`（`anchors`→`window`+`frames[].box`），用 `cli.run(return_track_samples=True)` 取逐帧 track 样本，算 `frag_count`（并集大小）+ `stationary_acc`。
  - `src/redlight/evaluation/module_metrics.py`：`iou_box`(L62)、`attribution_union`(L80)、`mask_iou` 可直接复用。
  - `scripts/gen_gt_skeleton.py`：生成 B2 骨架（window + 2 锚帧 ts，`box=None` 留 Jacob 标）。
  - `cli.run` L175-178：`return_track_samples=True` 返回 `engine._track_samples`（tid→[{ts,stationary,box,...}]），纯加法无行为改动。
- **缺口澄清**：9 个骨架已存在（`datasets/gt/tracking/`：违章02-09、11，window + 锚帧 ts 已填，`box:null` 待 Jacob 填）；缺口只是 **box 像素未标**，非文件不存在。GT-anchored 精版卡在 **Jacob 标框**（项目约定：wb 只造骨架，像素值 Jacob 标）。

### 1.2 两步并行：GT-free 预览（即时）+ GT-anchored 精版（等 Jacob）

**A. GT-free 预览（标"下界"，不阻塞，立即可出趋势报告）**
- ⚠️ **明确标注为"下界 / 代理指标"**：它用 `violation_events` GT 窗作"违章车"时间定位**代理**（不是真 GT 框）。指标只能反映"跟踪在已知违章窗内稳不稳"，不能当覆盖率真值。报告与所有产物必须打 `下界` 标签，避免被误读为模块真值。
- 用 `violation_events` GT 窗作为"违章车"时间定位代理，取引擎对该事件关联的 track（`ev["track_id"]`+`member_tracks`）的 `track_samples` 子序列。
- 算 3 个 GT-free 不稳定指标（均标"下界"）：
  - **ID 切换数**：该子序列里"主导 tid"随时间变化次数（主导=该帧 box 与事件代表框 IoU 最高/最近的 tid）。
  - **断裂数**：子序列里"无 attributed tid 在场"的连续间隙段数。
  - **覆盖率**：窗时长内"至少 1 个 attributed tid 在场"的时间占比。
- box IoU 跳过（无 GT 框）。产出**逐视频跟踪不稳定性预览（下界）**，先给 cc 看趋势；并与端到端 eval 的 `覆盖` 指标对照，量化"跟踪断裂→覆盖率掉→漏检"的链路。

**B. GT-anchored 精版（Jacob 标完 box 后跑；骨架不复生成）**
- ⚠️ **骨架已存在，禁止重生成**：`gen_gt_skeleton.py` / `eval_tracking.py` 代码脚手架已就绪（Plan v4 留的），不再重写或重跑生成。
- 1.2.1 流程：交 Jacob 按现有骨架格式标 box（仅像素值，wb 不代标）→ 直接跑 `eval_tracking.py`（读 `datasets/gt/tracking/{video}.json`）。标注前先与 Jacob 确认交互方式（脚本已支持 fill-in，不阻塞 wb）。
- 1.2.2 扩展 `eval_tracking.py` 指标（在现有 `frag_count`/`stationary_acc` 基础上加）：
  - **ID 切换数**：GT 窗内对违章车构建"每帧主导 tid"时间序（attr_set 中 box 与锚框 IoU 最高的 tid；attr_set = 窗内任一帧与锚框 IoU≥T 的 tid 并集，T 默认 0.3 用于关联），统计主导 tid 切换次数。
  - **断裂数**：时间序中"无 attributed tid"的连续间隙段数（车丢失后重编号）。
  - **box IoU**：锚帧处主导 tid 框 vs GT 锚框的 IoU（mean）。
  - **覆盖率**：GT 窗时长内"≥1 attributed tid 在场"的时间占比。
  - 保留 `frag_count`（并集大小）作补充维度。
- 1.2.3 跑 → 出**跟踪现状报告**（交 cc）：逐视频 4 指标 + 聚合；标注哪些视频掉链子、掉在哪（结合 track 位置/occ 推断 occlusion/out-of-frame/similar-car）；**量化"跟踪断裂"对端到端覆盖率的影响**（对比 `eval_violations` 覆盖指标，看断裂视频是否覆盖低/漏检）。

### 1.3 验收（cc 来验）
- B2 报告交付：逐视频 4 指标（ID 切换/断裂/box IoU/覆盖率）+ 断裂对覆盖率影响量化。
- GT-free 预览先出（不卡 Jacob）；GT-anchored 精版在 Jacob 标框后补。
- **只建度量，不碰 tracker.py 任何逻辑**（红线：先诊断后修，修不修看报告）。

---

## 2. 顺带：Phase 1.5 crosswalk v2 过延伸护栏（**暂缓——先诊断根因**）

> cc 裁定：decay 方案三处硬伤，不急着上。改为"先诊断 05/09 过延伸根因，再上安全方案"。本阶段**不写护栏代码**，只做只读诊断 + 出安全方案设计，待 cc 复核后再决定是否实施。

### 2.1 问题
- v2 `detect()` 用 `np.maximum(self._accum, resp)` 跨帧 running-max（L74-77）。`self._accum` 单调增长 → 长视频里**过期条纹响应不退场** → 05/09 的 episode 甩出 GT 窗（F1 未受损因 1:1 匹配容忍，但部署新视频是泛化隐患）。

### 2.2 ❌ decay 方案三处否决（先记清楚，避免再走）
- **① 空操作**：`np.maximum(self._accum*decay, resp)` 在 `resp` 持续非零处永远被 `resp` 顶回满值——若过延伸源是"持续存在的假条纹"（非瞬时），衰减根本不起作用，等于没改。
- **② 数值错**：`_accum` 长视频（~数千帧）乘 0.97 累积的边界易写错（`None` 首帧、帧跳变）；且 `band_thr=0.10·maxc` 依赖 `maxc` 同步衰减，否则阈值与响应不同步会让真斑马线在弱帧被误杀。
- **③ 伤抗遮挡**：真斑马线在遮挡帧 `resp` 骤降，衰减会让 `accum*decay` 跟着塌，遮挡结束瞬间 band 失稳 → 抗遮挡能力退化（v2 当前抗遮挡是卖点，不能伤）。

### 2.3 正确路径：先诊断再上"冻结或收紧百分位"（安全方案）
- **第一步（只读诊断，不写码）**：量化 05/09 过延伸的**真实根因**：
  - 是"跨帧 running-max（`_accum`）把单帧假条纹（路边栏杆 / 车顶亮条）永久化"？还是"分行百分位（5–95）在长 episode 把端帧 outlier 纳入 band"？
  - 抽帧 + 打印逐帧 `resp` 与 `_accum` 曲线，定位过延伸来自 `_accum` 不退场 还是 百分位窗口过宽。
- **第二步（诊断后设计，待 cc 复核）**：安全方案二选一，均不伤抗遮挡：
  - **冻结百分位**：band 在首段锁定后**不再更新（freeze）**，后续帧只 mask 不重算 band → 过期条纹进不来；真斑马线已被冻结在 band 内不受伤。代价：相机漂移场景需轻量重锁条件。
  - **收紧百分位**：把 5–95 收成 10–90（或更窄）→ 端帧 outlier 被排除 → 过延伸收敛；真斑马线主体不受影响。需 eval 实测不伤覆盖。
- 任一方案都先 TDD（合成过延伸序列断言收敛）+ 护栏双验证（F1≥0.889、8 好零回退、负例 0）→ 净回退不上。
- **未诊断前不动 `crosswalk_v2.py` 任何行**（红线：先诊断后修）。

---

## 3. 提交纪律（红线）
- `scoped git add <具体文件>`，**禁 `-A`**。
- 署名 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- trunk(`main`)。
- 不碰 GT 进生产、不加 per-video 硬编码、不动 `enforce_transition_limit`。

## 4. 明确不做（记录，免得又被卷进去）
- 灯态误绿手调门控 / 切 `method=ped_classifier` —— 暂停。ped_classifier 只接 `detect()`（非端到端 `observe()` 路径），且 36KB 模型训练数据=这 9 视频（过拟合）。它是下一个正经立项（接线 + 含负例/背心重训 + 验泛化），不是现在补丁。
- 04 短绿（1.2s 刀尖 fn）、01 负例误绿（1 fp）—— 都是灯态，随灯态立项解决，现在接受存在。

## 5. 待 cc 复核放行（已按本回合指令精炼）
- **B2 已放行**：本回合先实施 **A（GT-free 预览，标"下界"）**；B（GT-anchored 精版）等 Jacob 标框后跑，骨架不复生成。
- **Phase 1.5 暂缓**：本回合只做 §2.3 只读诊断 + 安全方案设计，待 cc 复核后再决定是否写码。
- **仅剩一个待决**：3 份门控 docs（plan-v7 / phase1-report / diag-gate-ineffective）是否 scoped add 入仓作"此路不通"历史（建议入仓，便于下次灯态立项直接读）；或保持 untracked。
