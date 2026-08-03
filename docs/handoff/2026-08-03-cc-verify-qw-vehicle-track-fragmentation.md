# CC 复核 qw 违章车 track 碎片化诊断(b48bb55)— 数据两层通过, Q2"结论对但论据错": 过度合并真实存在(cc 独立坐实机制), 但 qw 的"负缺口"论据无效(是锚 undercount)

> 出自 cc(arbiter)。qw 交违章车碎片化 GT-free 诊断(b48bb55), 头条 Q2 判"`_dedup` 过度合并、制造假合并 = F1 看不见的真问题"。cc 两层复核: **底层数据真实, 且过度合并确实存在(cc 用独立证据坐实), 但 qw 用来支撑它的"掩盖缺口=-6"论据站不住(是低覆盖率锚 undercount 伪影)。结论保留、论据更正、并揪出更完整的双向错误。**

## 0. 独立复核(两层)
- **聚合层**(cc 从 `vehicle_track_fragmentation_per_episode.csv` 自算, 不跑 qw 脚本): truefrag03 均值 24/13=1.8、truefrag05 15/13=1.2、masked 30/13=2.3、gap03 总和 24-30=-6(均值 -0.5)、coverage 均值 0.72(<0.7 计 6/13)、成因 a2/b2/c5/none2 共 11 对。**全部对上报告。**
- **重建层**(cc 自写脚本独立重跑 `cli.run("违章07", return_track_samples=True)`, 非 import qw 的 `diag_vehicle_track_fragmentation`): confirmed=3, ep0 masked=1/cov=1.000/true03=1、ep1 masked=2/cov=0.222、ep2 masked=2/cov=0.999 —— **逐 episode 复现 qw 的 CSV, 数据真实。**

## 1. ⚠️ Q2 的"负缺口=过度合并"论据无效(锚 undercount 伪影)
- **-6 几乎全来自 违章03 ep0 单个 episode(gap=-6)**。qw 自己的逐碎片 CSV 已坐实: 锚 track 68 `first_ts=91.461 / last_ts=110.992`, 而 episode 窗到 `137.8s` —— **锚在 111s 就死了, 后 27s(58% 窗口)锚缺席**。那 8 个 member_tracks 大半活在锚看不见的后半段, `_episode_fragments` 只在"锚在场的 ts"找重叠 track(`diag:92-99`)→ truefrag 天然数不到 → gap=-6 是**锚 undercount, 不是过度合并**(coverage 0.424=19.5s/46.3s 正好对上)。
- **按覆盖率分层**(cc): coverage≥0.7 的 7 个 episode, gap03 = 0/+1/0/0/0/-1/0 → **净和=0**。高覆盖率下掩盖缺口≈0, 负值全是低覆盖率伪影。
- **gap 指标本身有缺陷**: truefrag 与 masked 是**两个不同集合**, 相减把"过度合并(member 有、锚不重叠)"和"漏掉真碎片(锚重叠、member 没有)"两个**方向相反**的错混为一个数。07 ep1 就是活例(见 §2), 二者恰好抵消成 gap=0, 掩盖了两个都存在的错。**gap 的正负不能作为过度合并的判据。**

## 2. 但过度合并**确实存在** — cc 用独立证据坐实(结论保留)
cc 重跑 违章07, 对每个 member track 查它与锚的**时间重叠**+**IoU**(区分 undercount vs 真过度合并):
- **07 ep2**(cov 0.999): member `track 20` 与锚**整窗时间重叠(39/39 帧共存), 但 box IoU 恒 0.122<0.3** → 空间上**不同的框被 `_dedup` 合并进同一 violation episode**。真过度合并, 铁证。
- **07 ep1**(cov 0.222): member `track 13` 时间重叠(8 帧)但 **IoU=0.000** → 又一例。
- **机制**(承侦察读的 `violation_engine.py:254-298 _dedup`): **纯按时间重叠/间隔<gap 跨 track 合并, 不看空间/身份** → **不同车在重叠时间各自违规, 会被塌成一个 episode**。这不是"碎片被藏", 是"不同车被错并"。
- **同时另一个方向的错也真**: 07 ep1 的 truefrag 集合是 `{3, 17}`(track 17 与锚重叠、`in_member=0`)、masked 集合是 `{3, 13}` → track 17 是 `_dedup` **漏掉的真碎片**、track 13 是**过度合并的**。两个相反的错并存, gap=0 把它们抵消掉了。

## 3. 裁定 — 三问重新定性
- **Q1 违章车切几个 ID**: per-episode 真碎片 T=0.3 均值 **1.8(下界, 6/13 低覆盖率)**、T=0.5 均值 1.2。远低于 [[b2-tracking-fragmentation-blindspot]] 记的"5–45"—— 差异**基本是口径**: "5–45"应是**整段视频原始 track ID 总数**, 本诊断量的是 **per-episode 违章窗内锚重叠碎片**, 二者不可直接比。碎片化真实但 per-episode 温和。
- **Q2**(核心, 更正): ~~"负缺口=过度合并"~~ 论据作废。**改判: `_dedup` 存在两类真错 ——(A) 时间-only 过度合并把空间不同的 track/不同车塌进一个 episode(07 ep2 track20 IoU0.12 整窗共存铁证);(B) 也会漏掉部分真碎片(07 ep1 track17)。** 正确度量应走**集合归属**(member∖truefrag=过度合并候选、truefrag∖member=漏并碎片), **且只在高覆盖率 episode 上判**, 不能用 gap 正负。
- **Q3 成因**: c_duplicate=56%(11 对里 5 对)—— n 太小(且 3/5 来自 违章05 ep2 单个)且建立在锚受限的碎片集上, **仅供方向参考, 不足以定量**。

## 4. 战略含义(reframe 盲点)
- 项目里**更有后果的 tracking 问题不是"碎片被合并掩盖", 而是 `_dedup` 的时间-only 过度合并会把不同车/不同框塌成一个 violation episode** —— 这直接威胁违章**归属正确性**(可能张冠李戴车牌、错并两起违规为一起、或吞掉本应独立的 FP)。这是比碎片化更该立项的方向。
- 立项修法候选(**不在本任务做**): `_dedup` 合并时加空间/IoU 或 track-identity 约束(不能纯时间);碎片侧(A 类)则调 `max_disappeared`/`iou_thresh`/加 re-ID。需先补 `datasets/gt/tracking/*.json` 的 null 框(Jacob)才能上 `attribution_union` 精确量。

## 5. 对 qw 的反馈
- **正向**: 过度合并的**直觉对了**(确实存在), 逐碎片 CSV 出得全(让 cc 能坐实机制), coverage caveat 也报了。
- **要改**: **头条结论不该建立在自己报告已标"下界"的指标(gap)上** —— coverage<0.7 占 6/13 时, 先按 coverage 分层再下结论, 别让一个 undercount 的 episode(违章03)带节奏。且 truefrag 与 masked 是不同集合, 相减的语义要想清楚再当指标用。下次: 结论前先做敏感性/分层检查。

## 6. 红线
只读诊断已交, 数据/机制 cc 已独立坐实。修 `_dedup`/tracker、标 tracking GT 框均为后续 write 立项(assignable to wb/Jacob), 非本任务。cc 未改任何生产代码/GT(重跑仅读)。

---
**一句话**: qw 违章车碎片化诊断数据两层通过(聚合 + cc 独立重跑 违章07 复现)。**Q2 结论(`_dedup` 过度合并)对、论据(负缺口 -6)错**: -6 是 违章03 锚在 111s 死于 137.8s 窗的 undercount, 高覆盖率子集净 gap=0, 且 gap 把"过度合并"与"漏碎片"两反向错抵消。cc 用独立证据(member track 与锚整窗共存但 IoU<0.3)坐实过度合并真实存在(07 ep2 track20 IoU0.12、ep1 track13 IoU0), 机制=`_dedup` 纯时间合并不看空间 → 不同车塌成一个 episode。**这是比"碎片被掩盖"更该立项的违章归属正确性 bug**。碎片化本身 per-episode 温和(1.8 下界), "5–45"是 per-video 口径不可比。修 `_dedup` 加空间约束是后续 write 立项。
