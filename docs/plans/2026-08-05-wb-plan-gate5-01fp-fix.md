# 违章01 FP 修法 #3 时序门控方案 (plan-gate #5 预审稿)

> 本文件为**方案文档**, 不含任何生产码改动。cc 审 scope → 放行后 wb 才建 worktree / 写码, 按 G1–G4 交证据表。
> 授权来源: cc 独立 bit-for-bit 复现时序横测 (commit `693d2ca`; cc_temporal_verify 21:59 vs wb diag_temporal_sep 19:34 逐 bit 相同) → 轴A 活 / 轴B 死 → 可分性 CONFIRMED。Jacob 硬停止规则(两轴都不可分才 de-scope)触发"不 de-scope"分支 → 授权出 #3 时序门控方案, 走正式 plan-gate #5 (审方案, 非放行写码)。

---

## 0. 授权状态与职责边界
- 本文件 = #3 方案预审稿, 答 plan-gate #5 六硬条件。
- **授权 ≠ 放行写码**: 方案过 gate 后 wb 才改生产码; 本文件未改任何 `.py`。
- 红线全程适用(见 §6)。

---

## 1. 根因复述(一句话)
01 observe prior 直采 `_sample_roi`(traffic_light.py:626) 在遮挡期采环境绿 → 产出 **8 段碎绿 (raw 最长绿 run 仅 3.10s)** → `fuse_light`(temporal_fusion.py:33) 经 window=24 / hysteresis=0.68 / unknown_hold=8 + `enforce_transition_limit(max_transitions=2)` 把 8 段瞬态绿**桥接成 13s 单 green 段** → `decide_violations`(decision.py:35) 把该 green∩静止∩压线判为 **confirmed** → FP。

**关键洞察(轴A 实证)**: 判别量不是"融合后总时长"(01=13s 会骗人), 而是"段内**最长连续 raw 绿 run**"——真绿 ≥ 26.8s(07), 假绿 ≤ 3.10s, **8.6× margin**, 过 06 低饱和石(31.22s ≫ 3.10s)。

---

## 2. 机制: raw 绿 run 持续性门控 (纯时序判据)

### 2.1 改动面(最小, 两处)
- **A. 标注 — `temporal_fusion.py` `fuse_light`**: 函数内部已持有 `observations=[(ts,obs,conf)]` 与逐帧 `raw` 序列。对产出的每个 `green`/`flashing` 段, 计算该段时间窗内**最长连续 raw 绿 run** (`max_raw_green_run_s` = obs=="green" 连续帧折算时长, 遇 red/unknown/off 中断即重置), 附到 segment dict。
- **B. 门控 — `decision.py` `decide_violations` / `_go_intervals`**: 排除 `max_raw_green_run_s < T` 的 green 段, 改进 `review_light`(或新增 transient 桶 → review)。即: **瞬态绿不进 confirmed, 降级 review**。

### 2.2 不改的(红线)
- 不动 `_sample_roi` 外观 / 面积投票 / ROI 尺寸; 不动 `sat_min`; 不碰 `light_priors.json` 坐标与 `ped_signal.pt`; 不引入 per-video 灯参。
- 仅新增一个时序判据 + 一个 config 项 `traffic_light.min_persistent_green_run_s`(默认 10.0)。

### 2.3 阈值 T 与防过拟合
- 初值 **T=10.0s**: 落在 01 最长 3.10s(3.2× above)与真绿最短 26.8s(2.7× below)之间的 **GAP**, **不是拟合 01 单点**而是拟合两群间距。
- **中间态(raw 绿 run 5–10s)→ 降级 review, 非硬杀**: 即使 T 误设, 最坏结果是进 review 人工队列, 不静默丢绿 → 把"过拟合误杀真绿"的代价压到最低。
- T 最终以 G1(视频10 负例回归)+ G2(全 11 视频回归)后定稿, **不预锁**。

---

## 3. plan-gate #5 六硬条件逐条应答

| # | 硬条件 | 本方案应答 |
|---|--------|-----------|
| C1 | **n=1 过拟合(最重)**: 机制只由 1 个假绿定义"短"类; 中间态(raw run 5–10s)降级 review 非硬杀 | 阈值落两群 GAP 非拟合单点(§2.3); 中间态降级 review 兜底; 必补 G1 视频10 负例 + G4 公交/大车遮挡扫描 |
| C2 | **补负例 10 回归**(本横测未测 10) | G1: 用 `diag_temporal_separability.py` 跑视频10, 确认其 maxGreenRun 不与真绿重叠、改造后 01+10 均 0 FP。**方案放行前置证据门** |
| C3 | **公交/大车遮挡致真绿 raw run 碎短被误杀** | G4 专项扫描 05–09(+10)的"长绿段但其最长 raw run < T"实例, 确认是瞬态(应 review)而非应 confirmed 的真短绿; 若有真短绿被误杀 → 触发 T 上修或机制回退讨论。降级 review(非硬杀)已把 FN 代价压到最低 |
| C4 | **7 好视频全 TP 不回退** | G2 全 11 视频端到端重跑(7 好 + 05–09 + 01), 以 07 列为 F1 基准, 确认 confirmed TP 数不降、review 量不失控 |
| C5 | **只加时序判据, 不碰 prior 外观 / light_priors.json / ped_signal.pt** | 改动面限定 §2.1 的 A+B 两处; 红线清单 §2.2 明示排除项 |
| C6 | **声明局部止血非根治** | 方案中明示 #3 只消 01 单点 FP(止血), 真因(环境绿误读)根治归 `light-classifier-retrain` 判别器线, 不宣称根治 |
| C7 | **正向耦合 qw P2 车牌线**: #3 消 01FP 后, qw P2 在 01 上的连带错罚单(京N8ZK53)自动归零 | 建议 #3 先行于车牌线收口(cc 已认可方向) |

---

## 4. 证据门(plan-gate #5 放行前必须齐)
- **G1**: 视频10 负例回归(0 新增 FP; 若有既有 FP, 时序门控消之且不误杀)。**前置**。
- **G2**: 全 11 视频端到端 F1 不回退(07 基准)。
- **G3**: cc 亲测复核时序可分性表(已 done, `693d2ca` + cc_temporal_verify 21:59 逐 bit 同)—— 维持。
- **G4**: 公交/大车遮挡专项扫描(05–09/10)无真短绿被误杀。

---

## 5. 改动文件与回归清单
- 文件: `src/redlight/pipeline/temporal_fusion.py`(`fuse_light` 加 `max_raw_green_run_s` 标注) + `src/redlight/pipeline/decision.py`(`decide_violations`/`_go_intervals` 加门控)。**单点两处**。
- 配置: `configs/config.yaml` 新增 `traffic_light.min_persistent_green_run_s: 10.0`(不改既有项)。
- 回归: 复用 `cli.run` 逐视频(前台 ≤3/批)或现有回归脚本, 产 violations.csv 比对。

---

## 6. 红线合规声明
- 本文件为方案文档, 未写任何生产码; 过 gate 前不建 worktree/分支。
- 禁 `select_gtfree`; 不碰 `light_priors.json`/`ped_signal.pt`; 权重不入库; scoped git(禁 `git add -A`)。
- 诊断产物 gitignored(沿用 `693d2ca` 既定语义)。

---

## 7. 风险与未决
- T=10s 对视频10 是否安全 → G1 定; 若 10 出现真绿短 run 且无法区分 → 触发回退 / de-scope 讨论(硬停止规则的收口分支仍可用, 不 de-scope 仅因轴A 当前可分)。
- `fuse_light` 改造若影响既有 flicker/unknown 行为 → G2 回归验证(尤其 07 闪烁真绿)。

---

## 8. 交付与下一步
- 本方案文档 = plan-gate #5 预审稿。
- **球在 cc**: 复核六硬条件应答(C1–C7)→ 放行 / 送回。
- 放行后 wb 建 worktree → 按 §5 改码 → 跑 G1–G4 交证据表 → cc 亲测复核 → merge。
