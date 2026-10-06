# cc 第二关方案 gate 裁定:qw 车牌事件回填修复方案(3825f30)— **PASS,放行实施**

> 审核对象:`docs/plans/2026-08-04-qw-plate-fix-plan.md`(commit `3825f30`)+ `scripts/diag_plate_bucket.py`(串味修)+ `scripts/diag_plate_track_attr.py`(定位工具)
> 署名:cc(plan-gate/独立复核) — 遵 [[measurements-disagree-find-the-bug]]、[[eval-methodology-gap-overfit]]、[[qw-plan-execute-loop]] 第二关(方案 gate)
> 结论:**三硬条件全部 cc 亲验坐实(非采信 qw 口头),方案 PASS。放行 qw 进隔离 worktree 按 P2→P1→P3 实施,每步独立过 cc 效果 gate(plan-gate #3)。**

## TL;DR
1. **③定位(最承重)= bit-for-bit 复现**(上会话):`diag_plate_track_attr.py --video 违章02` → 事件 tid=26、member_tracks(17)=[2,26,34,37,44,50,49,64,69,68,78,86,87,89,93,94,97]、回填京A14672、tid 68 质心(87,263) 43.2-54.3s = 画面最左过路车,与 §0 逐项一致。
2. **①串味修 = bit-for-bit 复现**:`diag_plate_bucket.py --video 违章07` → `京EJQ505: [a_no_detect] 全视频无 ED<=2 读数(别车框不算该车证据)`,京ACG0878/京Q5D2N8 归 ok。跨车串味修真实。
3. **②P1/P2 合设 = 设计健全 + 数据可行 + 失效安全**(见 §3)。
4. **b2 过合并(17 tid 脏袋)= 上游 tracker 问题,方案用同一性判定绕开、不修根因,记独立立项交 Jacob**——scope 划分正确。

---

## 1. 硬条件①:bucket 跨车串味修复(cc 亲跑)

```
违章07 京ACG0878: [ok] ED=0 京ACG0878 conf=0.92 box=93x69 @f2496
违章07 京EJQ505:   [a_no_detect] 全视频无 ED<=2 读数(别车框不算该车证据)
违章07 京Q5D2N8:   [ok] ED=0 京Q5D2N8 conf=1.00 box=119x85 @f0
```
`bucketize` 改用文本关联(`near = [r for r in recs if levenshtein(r[1], gp) <= 2]`)判桶——别车框不再冒充该车证据。度量具就绪、不再高估。**PASS。**

## 2. 硬条件③:京A14672 来源定位(cc 亲跑,上会话)

`diag_plate_track_attr.py --video 违章02` 输出与 §0 逐项一致:事件 track_id=26(白车 x=615 路面区),member_tracks 17 个 tid,回填 plate=京A14672;京A14672 13 帧归属 tid 68=7 帧(质心(87,263),最左车道,43.2-54.3s)。**结论坐实:京A14672 是被 b2 过合并误并进 episode 的过路车牌(tid 68 ∈ member_tracks,consensus best weight 最高 → `_episode_plate` 回填)→ 修法落 P2/P1 同一性约束,不落 b2。PASS。**

## 3. 硬条件②:P1 与 P2 合设健全性(cc 读生产码坐实)

审的不是数字,是"member_tracks 归组绕开脏袋"是**真解**还是**把脏袋换地方吐**。读生产码坐实两条数据前提 + 一条失效安全:

- **bug 源坐实**:`cli.py:187 _episode_plate` 遍历 `[track_id]+member_tracks` 取 weight 最高牌,**无时间/空间约束**(L193-199)→ 脏袋里过路车牌 weight 最高即回填。与 §2 问题描述一致。
- **P2 数据前提(真加性)**:`plate_consensus.py` 现仅按 tid 记 `{text, conf, ts}`(L29-32),**无 box**。qw §2.4"在 `n_consensus`(dag.py:96-110)把 plate 框随 update 一起记录"是真加性改动——扩展 `update` 记录结构,不动 ED 分组投票语义(L44-66)。且 plate 读取**已在读时关联到 tid**(`update(track_id,...)` 就是关联),加 box 只为做空间交叉核对,不新建关联。概念干净。
- **P1 数据前提(现成)**:`violation_engine.py:154 _track_samples = tid -> [{ts, stationary, box, overlap, cls, conf}]` 已有**每 member track 逐帧 box+ts 轨迹**。P1 归组(空间连续+时间重叠)有真轨迹可用:tid 68(x=87)与代表 track(x=615)box 轨迹空间不相交 → **能真正分开,非把脏袋换地方吐**。且 §3.4 明确用 `_track_samples` 做时空聚类、**不改 tracker 本体**(b2 独立项)——scope 干净。
- **失效模式安全**:两车真交叠的歧义帧,P2 §2.3 宁缺毋滥返**空串**,绝不回退别车牌 → 空(待人工补)> 错(开错罚单)。最坏情况也不制造 FP。

**②PASS。** 设计健全、数据可行、失效安全。

---

## 4. 裁定:PASS,放行实施(附效果 gate 硬条件)

放行 qw 进**隔离 worktree**([[multi-agent-worktree-isolation]])按 **P2→P1→P3** 增量实施。每步独立 commit + 独立过 cc 效果 gate(plan-gate #3),硬条件:

1. **P2 是安全闸,先过**:主验收 = **违章02 不再回填京A14672(空牌)+ 其余 6 命中零回退 + 负例零新牌**。**误罚=0 是 P2 硬 gate**;P2 不要求提升命中率,只消除开错罚单。
2. **加性证明**:`plate_consensus` 记 box 的 diff 必须证 ED 分组投票语义**逐 bit 未变**(cc 亲 diff);P2/P1 **不改** `tracker.py`/`violation_engine.py` 判定/`plate.py` 识别核心。
3. **P1 归组只用 `_track_samples`,不碰 tracker**;每车组≤1 牌;过路车组(空间不连续)不产牌。验收 = 事件车牌命中率 6/12→9/12(07/08/09 各 +1),违章02 保持空牌。
4. **主指标 = 事件车牌命中率**(cc 背书,生产口径),次指标 = **误罚=0**;副产物 `eval_plate` 84.6% 不回退(P4 不投)。cc 效果 gate 会**亲跑复现**每步命中率表 + 误罚证明,不采信报告数字。
5. 红线沿旧:隔离 worktree / scoped add(**禁 `git add -A`**)/ qw 署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>` / GT 只进诊断不进推理 / 权重不入库(不碰 `ped_signal.pt`)/ 原子写(tmp+os.replace)。
6. **b2 过合并(17 tid 脏袋)= 独立立项交 Jacob**,本方案绕开不修根因——记账,别在本 scope 里悄悄改 tracker。

---

## 5. 一句话给 Jacob
qw 车牌方案三硬条件 cc 全亲验:③定位(京A14672=过路车 tid 68 被 b2 误并)bit-for-bit、①串味修(京EJQ505→a_no_detect)bit-for-bit、②P1/P2 合设(`_track_samples` 有 box+ts 真轨迹能分车、consensus 加 box 纯加性、宁缺毋滥失效安全)读码坐实。**方案 PASS,放行 qw 进隔离 worktree 按 P2→P1→P3 实施,每步过 cc 效果 gate,误罚=0 是 P2 硬闸。** b2 脏袋根因记独立立项待你排期。

---
*署名:cc(plan-gate #2/方案 gate/独立复核)。本裁定只读复核未改生产码;三硬条件证据=亲跑两诊断脚本 + 读 cli/plate_consensus/violation_engine 生产码。承 [[plate-line-diagnosis]] 诊断 gate、[[b2-tracking-fragmentation-blindspot]] b2 独立项。*
