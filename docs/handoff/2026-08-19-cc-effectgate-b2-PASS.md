# cc 效果 gate:b2 脏袋收窄(ds 实施)= PASS

> 署名:cc(arbiter) 转交:Jacob(拍板 merge);抄送 ds/qw/wb
> gate 对象:分支 `b2-narrowing`@`f752afc`(ds 接手 qw,commits `6315888` fix + `f752afc` docs;未 push/merge)
> cc 独立复现:自建 detached worktree `crosswalk-guard-ccgate-b2`@`f752afc`(symlink input_video/models/.venv),**未进 ds 活 worktree**;跑 cc 自己的 `eval_violations.py --detector v2 --occ-denom box`(权威)+ ds 的 `qw_b2_audit.py`(逐 episode)。

---

## 1. 五条硬条件 —— 全部 cc 独立复现 PASS

| 硬条件 | cc 独立证据 | 判定 |
|---|---|---|
| ①P=1.000 零新增 FP | cc `eval_violations.py` 全 11(含负例 01/10):**TP=8 FP=0 FN=1@04 / P=1.000 R=0.889 F1=0.941 / 真误报=0 碎片=0**,与基线 `f8de88c` 逐字节一致 | ✅ |
| ②逐 episode 审计 | cc 跑 `qw_b2_audit.py`:每 confirmed episode span+member+member_all+命中 GT 全打印,**无窗外 confirmed**,04 无 confirmed、01/10 零 FP | ✅ |
| ③08/09 多车窗单 episode | 08 confirmed=1(span[0.7-49.2],双牌 京ACD5358+京ACW6553 都在)、09 confirmed=1(span[0.7-106.3],双牌 京AC63971+京NNM526) | ✅ |
| ④阈值 200px/3s 扫描透明 | ds §3 全 11×9 组合扫描表(缓存离线),D200g3 行与正式跑吻合;cc 采信方法(union-find 独立实现,共享阈值不共享码) | ✅ |
| ⑤Fix A 语义不动 | diff 确认 `_absorb`/transient_green **逐字未改**;09 保持 **confirmed 非 review 降级**(端到端印证 Fix A 未被破坏) | ✅ |

**member 收窄(cc 复现,narrowed←all)**:02:2←17 / 03:5←15 / **05:5←22** / 06:4←10 / 07:5←10 / 08:6←21 / 09:7←45 / 11:3←5 → **Σ=37←145(−74%)**,与 ds 报告逐数吻合。

**车牌逐 episode(cc 复现,d0 修复验证)**:03 京ABV3428 / 05 **''(空)** / 06 京N2LE10 / 07 京Q5D2N8 / 08 京ACD5358(+京ACW6553) / 09 京AC63971(+京NNM526) / 11 京AFW1222(+京AC81321) / 02 ''。**与 BASE 逐字节一致,0 误罚。关键:05 京N541E6 未复活**(d0 `member_tracks_all` 透传修复成立——这正是 cc plan-gate 点名的"去合并暴露车牌污染"风险,ds 在实施中自捕 d0/d1 两 bug 并修复)。

## 2. cc 判定 b2 的真实性质(比 ds 报告更直白,供 Jacob 决策)

**b2-as-delivered 对生产输出是 by-construction 零改动:**
- `_narrow_members` 在 `_dedup` **之后**运行,**不拆 episode、不改窗口/状态** → F1/P 结构性不变(不是巧合)。cc plan-gate 头号风险(去合并→新记分 FP)被 ds 的"只窄 member 列表、不动 episode"设计从根上规避。
- cc grep 全 `src/`:窄化后的 `member_tracks` **无任何记分/车牌消费者**;车牌走 `member_tracks_all`(=原始全 member=BASE 行为)。∴ 窄化只影响碎片化指标(member 计数),对 F1 与车牌**零功能影响**。

**推论(诚实记账)**:
- ✅ 交付的是:碎片化指标收窄(145→37)+ `member_tracks_all` 基础设施 + 零回退。
- ❌ 任务书 §6 头号目标"车牌 span 绕行可收回"**未达成**,且 ds 证明它**本质不可达**:窄化会缩小 京N541E6 质心 span → 反而击穿 span 排除约束。**两信号(车牌靠全量+span 排污 vs 车组收窄=代表集)互斥。** 这是有价值的**负结果**:member 收窄这条路救不了车牌 span 绕行。

## 3. 裁定:效果 gate PASS

- **五条硬条件全部 cc 独立复现通过,零回退、0 误罚。批准 merge `b2-narrowing`→main(交 Jacob 拍板)。**
- **但请 Jacob 知悉性质**:这是**零输出改动的碎片化指标清理 + 基础设施**,不是车牌归属修复。价值 = 指标更准 + `member_tracks_all` 基建 + 证伪"窄化收回 span 绕行"。若认为"零功能改动不值得进 main",可选择保留为已验证负结果不 merge——两者都合理,cc 无异议,取决于是否有下游消费碎片化指标。
- **未决(§7 承接)**:①车牌 span 绕行收回须车牌侧独立设计(超 b2 scope);②09 左停车带 23-tid 链成 1 组(member 45→7 已达标,按时间细分为开卷后续);③`member_tracks` 语义变为"车组代表集",下游 `eval_tracking_gtfree`/`diag_vehicle_track_fragmentation` 的 frag_count 会变小(方向一致=更准物理车数)。

## 4. 红线合规

cc 未进 ds/qw 活 worktree(自建独立 detached gate worktree,symlink 只读数据);未 push b2 分支;基线 `021dfce`/`f8de88c` 未动。ds 署名合规(`Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`,ds 代 qw)。

---
*署名:cc(arbiter)。证据=cc 独立 gate worktree 全 11 真跑 `eval_violations.py`(P=1.000/F1=0.941)+ `qw_b2_audit.py`(逐 episode member 37←145 + 车牌 0 误罚 05 空)。承 `docs/handoff/2026-08-19-cc-plangate-b2-PASS-conditional.md`、`2026-08-19-qw-b2-effect-delivery.md`、[[b2-tracking-fragmentation-blindspot]]。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
