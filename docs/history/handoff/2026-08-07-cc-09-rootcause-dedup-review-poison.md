# cc 定因订正:09 回归真因 = `_dedup` review 优先规则毒化 confirmed(非 prior-misframe)

> 署名:cc(coordinator/arbiter)  转交:Jacob;抄送 wb(修)、qw(b2 意见回复)
> **订正对象:我自己**——`docs/handoff/2026-08-07-cc-postmerge-accurate-perf-09-regression.md`(`359d654`)把 09 归因为「45-tid 脏袋 + 偏框 prior 采碎绿 <6s / TP 本脆」。**扒生产段级数据后证明该归因错误**,此文订正。
> 证据:生产 v2/box `cli.run` 直探 09 —— (a) 段级 `light_segments` + `max_raw_green_run_s` + evidence;(b) `_dedup` 前后 raw_events。

---

## 1. 09 段级实况(生产 v2/box,cc 亲探)

| 段 | 状态 | max_raw_green_run_s | evidence | #3 判定 |
|---|---|---|---|---|
| [0.0, 4.18] | green | **0.943** | visible | <6 → 瞬态 → **review** |
| [4.18, 7.41] | red | — | visible | — |
| **[7.41, 106.26]** | green | **55.892** | **visible** | **≥6 → 合法 confirmed** |

**09 有一整段 55.9s 的 visible 持续绿**(非 inferred / 非 occluded / 非反光瞬态),**完全通过 #3**,GT 违章段 [11-72] 落在其中。→ **qw #2 假设(公交反光/推断绿被 #3 弥散绿误杀)证伪;我的偏框 prior 碎绿假设也证伪。09 的绿是真·干净·长绿。**

## 2. 真因:`_dedup` review 优先规则(violation_engine.py:269,300)

dedup 前 raw_events(T=6):**45 个 confirmed 子事件**(全压在 55.9s 绿上,多 track b2 碎片)+ **1 个 review 事件**(tid5 在 [0.67,4.18] 瞬态绿上,0.94s)。dedup 后:**1 个 episode,status=review,[0.67,106.26],45 members。**

```python
# violation_engine.py:_absorb (line 299-301)
# review 优先级高于 confirmed(安全侧交人复核)
if e["status"] == "review" or cur["status"] == "review":
    cur["status"] = "review"
```

**跨 track 合并成一个 episode,只要任一成员 review,整个 episode 记 review。** #3 在 0.94s 瞬态绿上造的那 1 个 review 事件,把 44+ 个建立在 55.9s 铁绿上的 confirmed **全部拖成 review**。**这是 #3 ↔ dedup 的交互 bug,不是灯态/prior 缺陷,不是 09 TP 脆弱。**

**为什么 #3 前没暴露**:「review 优先」在 #3 之前是对的——那时 review 只来自遮挡/unknown(D1 真不确定,理应安全侧优先)。#3 引入了**新一类 review(瞬态绿)**,它能与同窗口的强 confirmed 共存;旧安全规则未区分 review 来源 → 误伤。

## 3. 影响订正:不是「01 vs 09 两难」,是可修 bug

我上一篇把它写成「精度换召回、Jacob 拍板选边」——**订正:这是 bug,不是 tradeoff。** 修 dedup 后:
- 09 恢复 confirmed(压在 55.9s 铁绿,合法);01 FP 仍消除(见 §4 安全性);
- **tp8 / fp0 / fn1(04) → P=1.000 / R=0.889 / F1=0.941** —— 优于 #3 前(0.889)与现状(0.875),**鱼与熊掌兼得,无需回滚 #3、无需选边。**
- 车牌线连带恢复:09 的 confirmed 载体回来,京AC63971+京NNM526 两牌重新落在 confirmed 事件上(命中口径恢复)。

## 4. 修法 brief(wb,决策层)

**目标:#3 的「瞬态绿 review」不得毒化同 episode 内证据充分的 confirmed。** 二选一(wb 出方案 gate):
- **方案A(推荐,按语义)**:给 review 事件打**来源标签**——`occluded/unknown`(D1 真不确定,保持「review 优先」安全语义)vs `transient_green`(#3,不得压过共存 confirmed)。dedup 合并时:episode 内**存在 confirmed 核** → 仅 D1-review 可降级,transient_green-review 被 confirmed 吸收(不改 episode 状态)。
- **方案B(更简,按物理)**:dedup 不跨「中间隔红灯段」的绿事件合并——[0.67,4.18] 瞬态在红灯 [4.18,7.41] 之前,与红灯后的 [7.41,106.26] 是**两个绿周期**,本不该并进一个 episode。

**01 安全性(必须在 gate 复核)**:01 的 FP 是**纯瞬态绿(3.10s),无共存长 confirmed 核**——两方案都不会复活 01(方案A:无 confirmed 核可保;方案B:01 无红灯后长绿)。**修 dedup 不回退 #3 对 01 的消除。** 这是硬约束,wb 方案须逐 bit 证 01 仍 0 FP。

**验收锚(cc 效果 gate 全 11 视频,不再犯子集漏测)**:09 confirmed 命中 [11-72] + 01/10 负例仍 0 FP + 02/03/05/06/07/08/11 零回退 + 04 不受影响 → 端到端 **F1≥0.941 / P=1.000**。红线:改 dedup/#3 交互,不碰 prior/权重/`_sample_roi`;独立 worktree;不 push/merge,球回我。

## 5. 回 qw 的 b2 三点

1. **b2 主攻方向 = 支持**(采信 05 实锤:22 member / 14 stationary,同时污染车牌选择 [京N541E6 混入] + 事件成形)。**但 09 已证明:b2 过合并只是「放大器」,不是 09 review 的直接因**——直接因是 dedup review 规则(§2)。b2 修好会让 09 episode 不再吞 45 track(减小误伤面),但**不修 dedup 规则,09 仍会被瞬态毒化**。两件事都要做,**dedup 规则修在前(wb,快、根治 09),b2 episode 合并收窄在后(qw,根治脏袋)**。
2. **b2 实施建议采信**:episode 合并加「窗口内静止 + 压线 + 时空连续」三重约束合理;验收基准(车牌 span 绕行数字 + member 数 22→少数 + 碎片化指标)合理。
3. **排期采信**:b2 在 Jacob 对整体拍板后启动,与 #3/dedup 修错开(都在 tracker/事件层,避免叠加难归因)。**建议顺序:wb 先修 dedup(解 09)→ 我验收 → 再 qw 启动 b2(解脏袋根因)。**

## 6. 一句话给 Jacob

我上一篇说 09 是「脆弱 TP、prior 采碎绿、要在 01 和 09 之间选边」——**我扒了生产数据,那个归因错了,在此订正。** 真相:09 有一整段 **55.9 秒干净的持续绿**,本该 confirmed;是 `_dedup` 那条「只要有一个成员待复核、整个事件就记待复核」的老安全规则,被 #3 新造的一个 **0.94 秒瞬态绿**触发,把 44 个铁证 confirmed 全拖成了 review。**这是个可修的 bug,不是两难**——修完 09 回来、01 误报仍消除,**F1 到 0.941(P=1.000),比合 #3 前还好**。不用回滚 #3,也不用你选边。我已给 wb 出修法(两个方案,含 01 不复活的硬约束),修完我全 11 视频验收。b2 我支持 qw 主攻,但顺序放在 dedup 修好之后。

---
*署名:cc(arbiter)。订正 `359d654` 归因错误。证据=生产 v2/box 09 段级(55.892s visible 绿过 #3)+ dedup 前 45 confirmed / dedup 后 1 review episode。承 [[postmerge-f1-875-09-regression]](订正)、[[wb-01fp-temporal-plangate5-passed]]、[[b2-tracking-fragmentation-blindspot]]、[[measurements-disagree-find-the-bug]]。*
