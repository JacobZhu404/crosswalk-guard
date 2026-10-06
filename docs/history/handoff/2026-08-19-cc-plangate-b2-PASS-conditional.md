# cc plan-gate:qw b2 脏袋收窄方案 = PASS(带硬条件)

> 署名:cc(arbiter) 转交:Jacob;抄送 qw
> gate 对象:`docs/plans/2026-08-18-qw-b2-episode-narrowing.md`(qw `5648709`)
> cc 独立复现:生产 v2/box `cli.run(return_track_samples=True)` 跑 05/09,逐 member 复算 span/stationary/overlap/质心。

---

## 1. 诊断复现 = 确认(bit-for-bit)

| 视频 | qw 报 member | cc 复现 | 白车碎片链 | 判定 |
|---|---|---|---|---|
| 05 | 22 | **22** ✓ | tid1[0,7.7]→11[7.8,22]→19[19.5,26.8]→25[24.8,46.7]/26[25.1,59.7],cx258-544,时序近连续 | 确认 |
| 09 | 45 | **45** ✓ | 45 member 跨[0,106],cx 64-1862 高度分散 | 确认 |

过合并病理属实:05 episode=[0.7,64.5](GT[0-30]),09=[0.7,106.3](GT[11-72])。**qw 的 pathology 量化与修法方向(时空连续区分同车碎片 vs 过路车)cc 采信。**

## 2. cc 发现的核心风险(比 qw §5 更尖锐)——gate 硬条件来源

**当前过合并正在遮蔽潜在 FP。去合并会把「被吸收的无害 member」暴露成「独立记分的 confirmed episode」:**

- **05 右侧车群**:`tid17 cx=1478 stat=0.94 maxov=1.00 span[17.9,64.5]` + tid2/9/30/45 cx~1370-1480。与白车(cx~410)空间距 >1000px ≫ 200px 阈值 → **必被拆成独立车组**。它跨 [17.9,64.5] 远超 GT[0-30];若拆出独立 confirmed episode → **新 FP,P 掉出 1.000**。当前被吞进巨块算 1 TP、未扣分。
- **09 尾段 member**([72,106] 段,超 GT[11-72]):同理拆出即新 FP。

**根本张力**:b2 收窄 member(改善车牌归属)与去合并暴露搭车成员为记分 FP,是同一操作的两面。

## 3. 裁定:plan-gate PASS,带以下硬条件(并入 cc 效果 gate)

1. **P 必须保持 1.000(零新增 FP episode)** —— 主条件,比聚合 F1≥0.941 更严。去合并后**不得**在任何 GT 窗外产生 confirmed episode。特别审计:05 右侧 cx~1478 车群、05/09 尾段 member 拆出后**不得**成为独立 confirmed FP。
2. **逐 episode 审计(全 11 视频)**:效果 gate 交付**每视频每 confirmed episode 的 [span]+member+是否命中 GT**,证明无窗外 confirmed。不接受只报聚合 F1。
3. **多车违章窗保持单 episode**:08(两白车)/09(GT 两车)真实多车窗**不得**被拆成多 episode(否则制造碎片行为)。验 08 仍单 TP。
4. **阈值扫描透明(no silent caps)**:200px / 3s(gap_merge)必须在 11 视频上扫描,报最终取值 + 理由 + 敏感视频;不得静默固定。
5. **保 Fix A 语义不动**:只收窄 `_dedup` 合并判定,不改 transient_green 吸收逻辑(方案 §5.3 已承诺)。

## 4. 放行

- **状态:plan-gate PASS / 授权实施**。qw 独立 worktree(署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`),不 push/merge,球回 cc 效果 gate。
- **基线**:全 11 F1=0.941 / **P=1.000**(dedup 落地后,`021dfce`)。
- cc 效果 gate 将在独立 detached worktree 真实 11 视频 v2/box 复跑,按 §3 五条逐项验。

---
*署名:cc(arbiter)。证据=cc 复现 05(22 member/白车链)+ 09(45 member/尾段分散)。承 `docs/handoff/2026-08-18-cc-task-qw-b2.md`、[[b2-tracking-fragmentation-blindspot]]、dedup gate `f8de88c`。*
