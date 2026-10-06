# cc 合并后准确效果复盘 + 09 回归定因 + wb/qw 下一步 brief

> 署名:cc(coordinator/arbiter)  转交:Jacob
> 触发:Jacob「收关。当前整体效果如何了,准确的怎样,整理下一步给 wb 和 qw 的任务」
> 证据环境:merged `main`@6124e19,生产 `configs/config.yaml`(**v2/box**),`eval_violations.py --detector v2 --occ-denom box` 全 11 视频(含负例 01/10);09 因果由 `cli.run` 直探 T=0(无门控)vs T=6(#3)逐 bit 对照。

---

## 1. 准确的当前整体效果(生产 v2/box,全 11 视频)

```
=== 总体事件级(头条 1:1): P=1.000 R=0.778 F1=0.875 (tp=7 fp=0 fn=2) ===
车牌命中=4/6   真误报=0   碎片=0
```

- **TP=7**:02/03/05/06/07/08/11(全 cov 达标)
- **FN=2**:**04**(已知灯态 FN,历史遗留)+ **09**(#3 副作用新增,见 §2)
- **FP=0**:01/10 负例零 confirmed(C7 兑现,01 假绿误报已消)

**对比合 #3 前**(wiring-b1 口径 [[crosswalk-v2-b1-c4-passed]]):tp=8 / fp=1(01) / fn=1(04) → **F1=0.889**。
**#3 用 01(FP)换掉了 09(TP)**:F1 **0.889 → 0.875(−0.014 微退)**,但 **precision 0.889 → 1.000**。

### 这不是简单的「退步」——是精度换召回的操作点迁移
执法自动开单场景,**误开罚单(冤枉守法司机)的代价 ≫ 漏检**。#3 把系统推到 **P=1.000(11 视频含负例零错单)**,并把 09 这种「脆弱证据」路由到**待复核队列(review),不是静默丢弃**——C1 兜底按设计生效。严口径 eval 只数 confirmed 故把 09 记 FN,但运营上 09 进人工队列,不是垃圾桶。**我倾向不回滚 #3。**

---

## 2. 09 回归定因(生产逐 bit)

| min_run(T) | 09 事件 status | 端到端判定 |
|---|---|---|
| **0(无 #3 门控)** | **confirmed** [0.67,106.26] green overlap=1.0 | 命中 GT[11-72] = **TP** |
| **6(#3,merged main)** | **review** 同段 | 无 confirmed = **FN** |

**#3 是 09 降级的直接因**(生产 v2/box 亲证)。两条附加事实:
1. **09 的 confirmed 事件是 b2 过合并巨块**:单事件 `member_tracks` = **45 个 tid** 横跨全片 [0.67,106.26]——与违章05 的 22-member 脏袋同病([[b2-tracking-fragmentation-blindspot]])。
2. **09 生产绿是碎的**:#3 能降级它,说明生产 v2/box 下与该车静止∩压线相交的绿段**每段 raw run 都 <6s**。而 wb 的 `verify_gates_01fp.py` 对 09 报的是**一整段 55.892s**——**gate 脚本的分段 ≠ 生产 v2/box 分段**。这正是「gate 脚本 G2/G4 全 PASS,生产却回归」的机制根源。碎绿与 [[prior-misframe-rootcause]](09 prior IoU=0 偏框)吻合:偏框 prior 采到的是瞬态/碎绿,不是那段真持续行人绿。

---

## 3. 我的 gate 问责(不粉饰)

我在 retune 效果 gate 放行 #3 时,**G2 只跑了 06/07/11 子集,漏测了 09**;且 **G2 的度量(数「与 GT 绿重叠的绿段个数」)结构上看不见 confirmed→review 的降级**——#3 不删段,只改事件桶,段数守恒故 `no_regression=True` 恒真。两个缺口叠加 → 我签了 PASS 却没发现 09 掉了。**这是我的验收漏洞,记账在此,不甩给 wb。** 教训沉淀 [[measurements-disagree-find-the-bug]]:效果 gate 必须以**端到端 confirmed 事件表(全 11 视频)**为准,不能只信单点门脚本的中间量。

---

## 4. 下一步 brief

### wb(灯态/prior 线)—— 09 prior 重定位(根治,承 [[prior-misframe-rootcause]])
- **不改 #3**(机制正确:它正确识别碎绿为瞬态并降级)。目标是让 09 的**真持续行人绿在生产 v2/box 下被检成一整段长 run**,从而 09 在 T=6 下**合法重新 confirmed**(真绿 55.892s ≫ 6)。
- **诊断先行**:定位 09 生产绿为何碎(先验 ROI 偏框 → 采到瞬态反光/过路绿?)。这是 prior-misframe 线已立的活(05/06/09 IoU=0),09 现在有了端到端 FN 作为**验收锚点**。
- **交付验收锚**:09 在 T=6 生产 v2/box 下 confirmed 且命中 GT[11-72];01 仍 0 FP;06/07/11 真绿零回退。达成即 F1 回到 ~0.889 且 **保住 P=1.000**(比 #3 前更好)。
- 红线不变:不碰 `ped_signal.pt`/`sat_min`;prior 重定位走画框真值坐实,禁 select_gtfree;独立 worktree;不 push/merge,球回我逐 bit 验。
- **04**(唯一另一个干净 FN,[42-43])是灯态线历史遗留,与 09 同属你线,顺带列入但优先级低于 09(09 是我刚引入的、且有明确根因)。

### qw(车牌/斑马线线)—— 车牌已收官,下一立项候选
车牌线 P1/P2/P3 已 merge 收官(命中 7/12,真事件误罚 0)。三个独立立项待 Jacob 排期,我按**杠杆**排序推荐:
1. **b2 过合并脏袋(最高杠杆,跨线根因)**:05(22-tid)+ 09(45-tid)两个 confirmed 事件都是脏袋,它**同时**污染车牌选择(05 选到别车牌)**和**事件成形(09 一个巨块吞全片)。修 b2 = 一个根因解两条线的伤。[[b2-tracking-fragmentation-blindspot]]
2. **斜角车牌 seg 检测模型**(02/05 60° 物理极限,HyperLPR3 弱项)。
3. **Phase 2 斑马线 seg**(距 0.941 天花板;[[crosswalk-ceiling-finding]])。
- 归属提示:b2 是 tracker 层,历史标注「下一立项、非车牌线」;但它是当前 F1 的**共性根因**,若 Jacob 同意可作为 qw(或跨线)下一主攻。**排期与归属请 Jacob 拍板。**

---

## 5. 一句话给 Jacob

准确效果:生产 v2/box 全 11 视频 **F1=0.875(P=1.000 / R=0.778,tp7/fp0/fn2)**,FN=04(老)+09(#3 新引)。**#3 拿 01 的错罚单换了 09 这个 TP**——F1 微退 0.014,但换来**零错单(P=1.000)**,且 09 是进了人工复核队列而非丢掉。我查实 09 的 TP 一直很**脆**:它压在一个 45 车的 b2 脏袋块 + 偏框 prior 采到的碎绿上,#3 只是把这脆弱点暴露出来。**我签 #3 PASS 时漏测了 09,这是我的验收缺口,已记账。** 建议:**不回滚 #3**;wb 去把 09 的 prior 重定位(让真绿检成整段,09 合法回 confirmed,同时保住零错单);qw 车牌收官后,下一主攻建议打 **b2 脏袋**(05/09 共性根因,一修解两线)。是否回滚、b2 归属与排期,等你拍板。

---
*署名:cc(arbiter)。证据=生产 v2/box eval 全 11(F1=0.875)+ cli.run T=0/T=6 直探 09(confirmed→review 逐 bit)+ 45-tid 脏袋 dump。承 [[wb-01fp-temporal-plangate5-passed]]、[[prior-misframe-rootcause]]、[[b2-tracking-fragmentation-blindspot]]、[[crosswalk-v2-b1-c4-passed]]、[[measurements-disagree-find-the-bug]]。*
