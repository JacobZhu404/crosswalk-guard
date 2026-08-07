# cc → wb 执行 brief:#3 时序门控重调 T(承窄送回 29aa717)

> 收件:wb(灯态线)  转交:Jacob
> 前置:cc 效果 gate 窄送回 `docs/handoff/2026-08-07-cc-effectgate-wb-01fp-fix3-sendback-T.md`(commit `29aa717`)
> 署名:cc(coordinator/arbiter)
> **一句话:机制/代码/接线/G1 全 PASS 保留,只做「调一个数 + 修一条注释 + 重跑三道门」。不推翻实现,不碰 prior/权重。**

---

## 为什么送回(30 秒回顾)

我授权你出方案时复核的可分性表只含 6 视频(01/05/06/07/08/09),真绿最短 26.8s → 8.6× margin。**那个子集不含违章11。** 你跑全 11 视频暴露:**违章11 的真绿本身就是一段短绿(单段 raw run 10.266s)**,全集真绿地板从 26.8s 塌到 **10.27s**。真实 GAP = **(3.10, 10.27)**,T=10 坐在其 97% 处——距真绿地板仅 **0.27s = 2.7% margin**,而其余真绿 ≥146%。违章11 是硬救回的 TP,其全部 confirmed 压在这一段上,检测抖动稍压到 <10 就翻 review = 丢 TP。这是 C1(n=1/子集过拟合)在边界应验。你把 buffer 给了假绿侧,真绿侧几乎零 margin。

---

## 要做的 4 件事

### ① 重调 T:10.0 → **6.0**
- 真实 GAP = (3.10, 10.27);几何中点 √(3.10×10.27)=**5.64**;取 **6.0**(01 距阈 **−48%**、11 距阈 **+71%**),两侧 margin 均 >30%。
- 硬约束:T 必须 **above 全部已观测假绿**(01=3.10 / 09=0.94)且 **below 全部真绿**(min=10.27)。区间 [5.5, 7] 均可,你可在此区间内据 G1/G2 数据定稿,但须附判定表说明为何取该值(不接受「沿用初值」)。
- 改动点:`configs/config.yaml` 的 `min_persistent_green_run_s` + 报告口径。**decision.py / cli.py / engine 的默认值也一并对齐**(别留 10.0 硬编码散落各处)。

### ② 修 config.yaml 假注释
- 现注释写「真绿最短(**26.8s**)」= 假,与你报告 §1 自陈的 10.27s 自相矛盾。
- 改为:真绿最短 = **10.27s(违章11)**,假绿最长 = 3.10s(01),T 落 GAP 中部,margin 据新 T 如实写。

### ③ 新 T 下重跑 G1 / G2 / G4(脚本已就绪)
用现成 `verify_gates_01fp.py`,把 fixed 轮的 min_run 从硬编码 10.0 改为受验 T,证:
- **G1**:01 confirmed 1→0(降级 review);09 假绿仍降级;video10 无新增 FP。
- **G2**:全 11 端到端,7 好视频 + 违章11 confirmed TP **零回退**;report review 桶增量(别让真绿冲爆人工队列)。
- **G4**:逐视频报每个真绿段 max_raw_green_run_s,证全部 ≥ T 且 margin 健康(违章11 应从 2.7% 抬到 ~71%)。

### ④ 口径统一
报告 + config 注释所有数字统一到 **全 11 视频**,删掉所有残留的 6 视频子集数(26.8s / 8.6× margin)。

---

## 红线(不变)
- 只加时序判据。**不碰** `_sample_roi`/`sat_min`/`light_priors.json`/`ped_signal.pt`/`enforce_transition_limit`。
- 禁 select_gtfree;GT 仅段级当裁判,不进推理。
- 独立 worktree(遵 multi-agent 隔离);scoped git add(禁 `-A`);按你线既有约定署名。
- 不 push / 不 merge,改完球回我,我亲测复现 G1/G2/G4(尤其违章11 margin + 01/09 降级)再放 merge。

## 保留(别重做)
机制(`_raw_green_runs`/`_max_run_in_segment`/瞬态绿→review 非硬杀)、代码结构、engine/cli/config 接线、G1 已验事实、`verify_gates_01fp.py`——全部保留。

---
*承 [[wb-01fp-temporal-plangate5-passed]]、[[crosswalk-ceiling-finding]](11 是硬救回 TP)、[[measurements-disagree-find-the-bug]]。合入后 qw 车牌线在 01 上的连带错罚单(京ADF5307)自动归零,故 wb #3 收口顺序先行。*
