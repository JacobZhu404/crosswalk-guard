# wb 报告: 违章01 FP 时序轴可分性横测 (brief 33965cf, 最后一轮)

> 承 plan-gate #3(:87 误绿, 8f9c152) → plan-gate #4 外观路线送回(c09d425) → 结构横测双死(8d94600)。
> Jacob 拍板最后一轮:**只测两条时序正交轴, 不预锁, 谁能留 margin 分开 + 过 06 低饱和石就用谁**;
> **硬停止规则**: 两轴都不可分 → de-scope 01 FP(同 04 FN)并入 retrain 线, 不再开新轮。
> 本脚本: `scripts/diag_temporal_separability.py`, cc 可 bit-for-bit 复现那张表。

---

## 0. 方法(沿前两轮纪律)

- 复用生产入口 `cli.run`(零循环复制), 仅用**只读** monkeypatch 抓逐帧状态:
  - `TrafficLightDetector.observe`: 只读复算 prior ROI 绿斑质心 `g_cx/g_cy`(相对 ROI 归一化)。
  - `BatchViolationEngine.accumulate`: 抓 **raw observe 层逐帧 `obs`**(融合前!) + `g_cx/g_cy`。
- **禁用 select_gtfree**; GT 仅裁判不进推理; 横测 01 假绿 vs 05/06/07/08/09 真绿。
- 轴A 只用 **raw observe 序列**(融合前), **绝不**用融合后 `light_segments` —— 这是防栽坑②(13s 持续陷阱)的硬约束。
- 红线: 只读, 不写生产码, 不建 worktree; 诊断产物 gitignored。

## 1. 可分性表(全 6 视频, cc 亲测复核)

| video | label | maxGreenRun(s) | nRuns | maxRedGap(s) | maxStepDrift | meanStepDrift |
|-------|-------|---------------|-------|--------------|--------------|---------------|
| 违章01 | false_green | **3.10** | **8** | 18.18 | 0.852 | 0.044 |
| 违章05 | true_green | 64.65 | 1 | 0.00 | 0.774 | 0.007 |
| 违章06 | true_green | 31.22 | 4 | 4.19 | 0.653 | 0.020 |
| 违章07 | true_green | 26.80 | 5 | 34.34 | 0.978 | 0.032 |
| 违章08 | true_green | 36.50 | 3 | 0.94 | 0.839 | 0.026 |
| 违章09 | true_green | 55.89 | 5 | 3.10 | 0.971 | 0.012 |

## 2. 两轴判定

### 轴A — 绿持续性 / 闪烁 (raw 层) = **活** ✅

- **主判别 = 绿 run 持续性**(`max_green_run_s`): 01 最长绿 run=**3.10s**, 真绿最短=**26.80s**(07)。
  **01 比真绿最短还短 8.6 倍, 留巨大 margin**。01 的绿碎成 **8 段**(瞬态), 真绿即使最碎(07 五段)每段仍 ≥26.8s(持续)。
- 物理吻合 brief 立足点: 过路车/反光是瞬态 → raw 层断续短绿 burst; 真信号绿是持续 → 长 run。
- **过 06 石**: 06 真绿 maxGreenRun=31.22s ≫ 01 的 3.10s, 低饱和真绿安全。
- ⚠️ **红缝隙不单独做判别**(见防栽坑①): 07 的 maxRedGap=34.34s 反而 ≫ 01 的 18.18s —— 真绿视频也有长红缝隙(红灯期)。所以红缝隙只作解释性参考(解释"为什么融合会把 8 段碎绿桥接成 13s 单段"),**不作为分离硬条件**。

### 轴B — 绿斑位置稳定性 = **死** ❌ (防栽坑③应验)

- 用 **maxStepDrift**(最脆弱维度): 01=0.852, 但 **07=0.978、09=0.971 ≥ 01** → 真绿视频(机位更抖)漂移反而更大, **无 margin**。
- 诚实附注: 若改用 **meanStepDrift**, 01=0.044 高于全部真绿(≤0.032), 该维度其实能分; 但 mean 维度被**偶发大幅抖动一帧**抬升 max(07/09 的 max 来自单帧跳变, 非持续漂移), 脆弱不可靠。**按"留 margin 分开全部真绿"的硬条件, 轴B 整体不可采用**。这正是 cc 预警的"手持漂移混淆"。

## 3. 三防栽坑点逐条验证(全部诚实作答)

1. **真绿是否也闪烁** —— **是, 且比 01 更长**: 07 maxRedGap=34.34s ≫ 01。已用此实证把红缝隙降级为参考指标, 主判别改用绿 run 持续性(不受影响)。若不量 07, 会误以为红缝隙能分 → 第五次栽坑被挡下。
2. **13s 持续陷阱** —— **全程只认 raw 层**: 表中 maxGreenRun 全来自融合前 `obs` 序列; 01 在 raw 层是 8 段碎绿(最长 3.10s), 融合后才被桥接成 13s 单段。若误用融合后层, 会把瞬态看成持续 → 假信号。已规避。
3. **手持漂移混淆** —— **已发生, 轴B 因此死**: 07/09 maxStepDrift ≥ 01, 证明不能把全局机位抖动记成灯漂移。已如实标记轴B 死亡, 不强行采用。

## 4. 结论 → 触发硬停止规则的哪一支?

**轴A 可分(绿 run 持续性, 8.6× margin, 过 06 石) → 不触发 de-scope。**

按 Jacob 硬停止规则("**两轴都**不可分才 de-scope"): 至少一轴可分即走修复路线。
→ **出 #3 时序门控方案, 走正式 plan-gate #5**; 01 FP 不 de-scope。

## 5. #3 时序门控方案方向(仅方向, 待 plan-gate #5 定 scope)

- **机制**: 在融合层(或 observe 层聚合)加 **raw 绿 run 持续性门控**——单段 raw 绿 run 过短(瞬态, 如阈值设在 **10s**, 留足 margin: 01 最长 3.1s 拒绿 / 真绿最短 26.8s 保留)则不提交为持续 confirmed green, 降级 unknown→review(:90)。
- **不改**: prior 直采外观阈值(四路已死)、`_sample_roi` 饱和/形状、`light_priors.json`/`ped_signal.pt` 坐标/权重。只加时序持续性判据, 非单帧外观阈值。
- **最小改动面**: 在 `fuse_light` 或 `decide_violations` 的 green 段提交前, 校验其 raw 支撑是否为持续绿(连续 raw 绿 run 时长 / 红 gap 占比)。

### plan-gate #5 放行硬条件(写码前必须全过)
1. **01 消误绿段**: 门控后 01 的 confirmed 违规归零(或转为 review)。
2. **真绿回归不杀**: 05/06/07/08/09 真绿帧保留 —— 阈值是安全的(真绿每段 ≥26.8s ≫ 10s); 尤其 07 已实测有四段长绿 run, 不会被误杀。
3. **端到端 F1 不回退**: 全 11 视频回归。

## 6. 红线合规

只读 · 未写生产码 · 未建 worktree/分支 · 禁 select_gtfree · 用生产可得信号(非 oracle) · GT 只裁判不进推理 · 诊断产物 gitignored · 未碰 `light_priors.json`/`ped_signal.pt`/权重。

---

*署名: wb(只读诊断)。交付: 本报告 + `scripts/diag_temporal_separability.py`(数据 `data/output/diag_temporal_sep/diag_temporal_sep.json`)。cc plan-gate #5 亲测复核可分性表。*
