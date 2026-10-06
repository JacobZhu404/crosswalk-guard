# cc plan-gate #5 裁定:wb 违章01 FP 时序可分性(693d2ca)— **可分性 CONFIRMED → 授权出 #3 时序门控方案**

> 审核对象:`693d2ca`(wb)— `scripts/diag_temporal_separability.py` + `docs/reports/2026-08-05-wb-01fp-temporal-separability.md`
> 署名:cc(plan-gate/独立复核)— 遵 [[measurements-disagree-find-the-bug]]、[[qw-plan-execute-loop]] 方案 gate、[[prior-misframe-rootcause]]、[[light-classifier-retrain]]、Jacob brief 33965cf 硬停止规则
> 复核环境:cc 主 worktree(main),独立 `--output data/output/cc_temporal_verify`(不覆盖 wb 的 `data/output/diag_temporal_sep`),端到端亲跑 6 视频。
> **结论:轴A(raw 层绿 run 持续性)可分性 cc bit-for-bit 复现;过 06 低饱和石;轴B(绿斑质心漂移)诚实判死。按 Jacob 硬停止规则「两轴都不可分才 de-scope」→ 至少一轴可分 → 不 de-scope → 授权 wb 出 #3 时序门控方案,走正式 plan-gate #5 审方案。**

---

## 1. 复现结果(cc 亲跑,bit-for-bit)

| video | label | maxGreenRun(s) | nRuns | maxRedGap(s) | maxStepDrift | meanStepDrift | vs wb |
|-------|-------|---------------|-------|--------------|--------------|--------------|-------|
| 违章01 | false_green | **3.10** | **8** | 18.18 | 0.852 | 0.044 | ✅ 一致 |
| 违章05 | true_green | 64.65 | 1 | 0.00 | 0.774 | 0.007 | ✅ 一致 |
| 违章06 | true_green | 31.22 | 4 | 4.19 | 0.653 | 0.020 | ✅ 一致 |
| 违章07 | true_green | 26.80 | 5 | 34.34 | 0.978 | 0.032 | ✅ 一致 |
| 违章08 | true_green | 36.50 | 3 | 0.94 | 0.839 | 0.026 | ✅ 一致 |
| 违章09 | true_green | 55.89 | 5 | 3.10 | 0.971 | 0.012 | ✅ 一致 |

cc 独立 JSON(`cc_temporal_verify/`,21:59)vs wb 提交 JSON(`diag_temporal_sep/`,19:34)**逐 bit 相同**(`diff` 归一化后 IDENTICAL,不同文件同内容=真独立复现非重读)。

## 2. 三条防栽坑点独立核验(cc 采信 wb 全部诚实作答)

1. **raw 层(防坑②「13s 持续陷阱」)= cc 亲验为真**。这是本闸的命门,cc 逐层坐实:
   - `dag.py:94` `ctx["light_observation"]=tl.observe(...)`,`dag.py:140-142` `engine.accumulate(..., ctx.get("light_observation",{}), ts)`——observe 与 accumulate 之间**无融合**,accumulate 抓到的 `obs` 就是 raw 逐帧观测。
   - `traffic_light.py` `observe()` docstring(:141)+ 实现:**单帧无状态**,不 append global_recent、不跑 `_state_from_global`;`global_recent`(maxlen=24)只在 `detect()` 用。
   - 脚本 monkeypatch `accumulate` 抓 `light_observation["obs"]` = 融合前。表中 maxGreenRun 全来自 raw;01 raw 是 8 段碎绿(最长 3.10s),融合桥接后才成 13s 单段。**若误用融合后层会把瞬态看成持续 → 假信号,wb 规避正确。**
2. **红缝隙不作硬判别(防坑①)= cc 采信**。07 maxRedGap=34.34s ≫ 01 的 18.18s——真绿视频(红灯期)也有长红缝隙。wb 自己抓到这条、把红缝隙降级为解释性参考、主判别改用**绿 run 持续性**(不受红缝隙混淆)。这正是 wb 亲手证伪自己 brief 里一个隐藏坑,记一功。
3. **手持漂移(防坑③)= cc 采信轴B 判死**。maxStepDrift:07=0.978、09=0.971 **≥** 01=0.852,真绿(机位更抖)漂移反而更大,无 margin。wb 诚实附注:meanStepDrift 维度 01=0.044 > 全真绿(≤0.032)看似能分,但被单帧跳变抬 max、脆弱不可靠 → **按「留 margin 分全部真绿」硬条件,轴B 整体不可采用**。cc 认同不强采。

## 3. 判定依据

- **轴A 活**:01 最长绿 run **3.10s** vs 真绿最短 **26.80s**(07)= **8.6× margin**。01 绿碎成 8 段(瞬态),真绿即使最碎(07 五段)每段仍 ≥26.8s(持续)。物理吻合:过路车/反光=瞬态短绿 burst;真信号绿=持续长 run。
- **过 06 石**:06 真绿 maxGreenRun=31.22s ≫ 01 的 3.10s——[[light-classifier-retrain]] 反复强调的低饱和真绿(最难保)安全。
- **硬停止规则**(Jacob brief 33965cf):两轴**都**不可分才 de-scope 并入 retrain。现轴A 可分 → **不触发 de-scope**。

**裁定:可分性 CONFIRMED。授权 wb 出 #3 时序门控方案。**

---

## 4. #3 方案在 plan-gate #5(审方案)必须回答的硬条件

授权≠放行写生产码。wb 的 #3 方案文档须在正式 plan-gate #5 前逐条答清,cc 才审:

1. **n=1 假绿的过拟合风险(最重)**:整个机制只由**一个**假绿视频(01)定义「短瞬态」类,只由 5 个真绿定义「长持续」类。10s 阈值 margin 虽大(3.1 vs 26.8),但样本 n=1。方案须论证阈值不是对 01 过拟合——至少给出:若未来出现「raw 绿 run 5-10s」的中间态假绿/真绿,降级到 review(:90)而非直接杀,是否安全兜底。**降级 review 优于硬拒,保住可召回性。**
2. **负例 10 必须进回归**:本横测**没测负例 10**(无违章)。#3 方案的 F1 回归必须含 01 **和** 10,证门控不在 10 上引入新行为。
3. **公交/大车遮挡致真绿 raw 碎裂**:方案须检查——真绿信号灯被过往公交/大车周期性遮挡时,raw 绿 run 是否会碎成 <10s 段而被误门控杀真绿?至少在现有 7 好视频 + 5 真绿上验证无一真绿段的 raw 支撑跌破阈值触发降级。若存在,阈值或聚合窗口须调整。
4. **7 好视频全 TP 不回退**([[crosswalk-v2-b1-c4-passed]] 基线):门控接在 `fuse_light`/`decide_violations` green 段提交前,须证对 7 好视频零影响。
5. **最小改动面 + 只加时序判据**:不碰 prior 直采外观阈值(四路已死)、`_sample_roi` 饱和/形状、`light_priors.json`/`ped_signal.pt`(遵 [[prior-misframe-rootcause]] prior 偏框根因线未结、遵红线不碰权重)。只加 raw 绿 run 持续性判据。
6. **与 retrain 线的边界**:治标(时序门控消 01 误绿)与治本([[light-classifier-retrain]] 判别器重训)不冲突,但方案须声明这是**局部止血**,不替代 retrain;01 只是当前唯一可复现的 prior 直采假绿,retrain 线仍是根治。

## 5. 附带提醒(交 Jacob + wb)

- 01FP 一旦被 #3 门控消掉,**qw 车牌线的连带误罚也随之消失**:qw P2 现把 01FP 事件回填 `京N8ZK53`=完整错罚单([[01fp-falsegreen-fix1-disproved]]、[[qw-p2-plate-effectgate-fail]])。#3 落地后 01 不再是 confirmed 事件,该错罚单自动归零。两线有正向耦合,收口顺序建议 wb #3 先行。
- 本裁定只放行「出方案」。写码前 plan-gate #5 审方案通过是硬前置。

## 6. 一句话给 Jacob

wb 的 01FP 时序可分性最后一轮:**轴A(raw 层绿闪烁持续性)cc bit-for-bit 复现,01 最长绿 run 3.1s vs 真绿最短 26.8s,8.6 倍 margin,还过了 06 低饱和石;轴B(绿斑漂移)wb 自己诚实判死(真绿机位更抖漂移反而更大)**。按你的硬停止规则「两轴都不可分才 de-scope」,现在至少一轴可分 → **不 de-scope,授权 wb 出 #3 时序门控方案**。我给方案钉了 6 条硬条件(最重是 n=1 假绿的过拟合、必须补负例10 回归、查公交遮挡会不会把真绿 raw 碎成短段误杀),方案文档答清这些、走正式 plan-gate #5 审过,才放行写码。另:#3 消掉 01FP 后,qw 车牌线在 01 上的连带错罚单(京N8ZK53)也会自动消失。

---
*署名:cc(plan-gate/独立复核)。证据=cc 主 worktree 端到端亲跑 6 视频 + JSON bit-for-bit diff + raw 层三处读码坐实(dag.py 接线 / observe() 无状态 / monkeypatch 抓融合前 obs)。承 Jacob brief 33965cf、[[measurements-disagree-find-the-bug]]、[[light-classifier-retrain]]、[[prior-misframe-rootcause]]、[[01fp-falsegreen-fix1-disproved]]。*
