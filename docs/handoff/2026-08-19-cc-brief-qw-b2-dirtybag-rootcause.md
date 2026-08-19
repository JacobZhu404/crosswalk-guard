# cc → qw brief:b2 脏袋过合并根因线(只读诊断先行)

> 署名:cc(arbiter → 派单 qw) 抄送:Jacob/wb/ds
> 触发:Jacob 采纳 cc 09 prior 重定位裁定(`f8fa121`)——09 归"接受不解降优先级",判别器线维持 HOLD,**让位 b2 线**。本 brief 启动 b2 线 #1 高杠杆项。

---

## 1. 上文(你接手前的定局,勿重开)

- **09 prior 重定位 = 证伪 OUT**(`docs/handoff/2026-08-19-cc-plangate-09-prior-relocate.md`)。重定位不修反恶化(GT=red 假绿 0.856→0.969),09 现 TP 是巧合但成立、无记分 FP。**09 不再动**,勿碰 prior/判别器。
- **判别器线 HOLD**:09 结构性无解已双重独立坐实;治本方向(若将来做)= 弥散绿空间/形状判别,非挪 prior、非重训 `ped_signal.pt`。**这条你也不碰。**
- **ds 交付的 b2(`_narrow_members`,已 merge `c51a121`)= 零输出改动的碎片化指标清理**,它在 `_dedup` **之后**运行、不拆 episode、不改状态。**它没修过合并本身。** 你这条线要修的正是它绕过去的那个根。

## 2. 你的线:`_dedup` 过合并("脏袋")根因 —— 为什么是 #1 高杠杆

`_dedup`(`src/redlight/pipeline/violation_engine.py` 约 :254-298)**纯按时间重叠/间隔<gap 跨 track 合并,不看空间/身份** → 时间上重叠的不同车/不同框被塌进同一个 episode("脏袋")。这是 **02/05 事件脏袋 + 碎片化 + 车牌误罚**的共同根因(车牌 span 绕行只是止血,已证 member 收窄救不了 span,见 [[b2-tracking-fragmentation-blindspot]] 2026-08-19)。

**cc 已独立坐实的铁证(2026-08-03,勿重测,作你起点)**:违章07 ep2 的 member `track20` 与锚**整窗 39/39 帧共存但 box IoU 恒 0.122<0.3**(空间不同的框被并入);ep1 `track13` IoU=0。→ 空间无关的时间合并把无辜车扫进违章事件,威胁**违章归属正确性**(张冠李戴车牌 / 两起并一起)。

## 3. 第一步 = 只读诊断(先诊断后动,禁写生产码)

在你自己的 worktree/clone 里跑,产出诊断报告交 cc plan-gate。要回答:

1. **全 11 视频量化 `_dedup` 跨 track 合并**:每个 confirmed episode,把 member 分成 **A=同车重编号**(空间一致、IoU 高或质心近)vs **B=异车误并**(时间重叠但 box IoU<0.3 / 质心远)。这是脏袋的"脏"在哪。
2. **每个 B 类误并单独判**:被误并进来的那辆车,是**真违章车**(拆出来该是合法独立 episode=拆了涨 TP)还是**旁观车/搭车**(拆出来 = 新记分 FP)?这是修法安全性的命门。
3. **车牌误罚机制**:把违章02 开错罚单 bug 链到 B 类误并(脏袋里混入别车 → 回填抓错牌),机制级坐实(承 [[plate-line-diagnosis]])。
4. **修法草案**:`_dedup` 合并加**空间/IoU 或 identity 约束**(不能纯时间);明确它对 §2 铁证(07 track20 IoU 0.122)与 §3.2 的效果。

## 4. 方法学护栏(踩过的坑,别重踩)

- **set-attribution 且只在高覆盖率 episode 判,禁用 gap 正负**(2026-08-03 教训:qw 头条"负缺口=过合并"结论对但论据错——gap 正负把"过合并"与"漏并碎片"两反向错抵消)。见 [[measurements-disagree-find-the-bug]]。
- **精确量化恐需 Jacob 补 `datasets/gt/tracking/*.json` 的 null 框**才能上 `attribution_union`——**卡住就明说、球回 Jacob**,别用软阈值硬凑。
- **测量打架就找 bug**,别折中别选边。

## 5. ⚠️ 修法的核心风险(plan-gate 会卡这条,现在就想清楚)

ds 的 b2 之所以零风险,是因为它**只窄 member 不拆 episode**——从根规避了"去合并暴露搭车 member 成独立记分 FP"。**你这条线如果真去拆脏袋,这个 FP 风险就被重新打开了。** 所以 §3.2(每个 B 类误并是真违章车还是搭车)不是可选诊断,是修法能否成立的前提:只拆 B 类里的**真违章车**(涨 TP),**误拆搭车 = 新 FP = 破 P=1.000**。

## 6. plan-gate 硬条件(提前告知,方案照此对齐)

`_dedup` 在 episode 形成的**最上游**,爆炸半径最大。方案/效果 gate 会要求:
- ① **P 保持 1.000,零新增记分 FP**(比聚合 F1 更严);
- ② 逐 episode 审计全 11(span+member A/B 分类+命中 GT,证无窗外 confirmed、无搭车被拆成 FP);
- ③ 车牌全 11 与 BASE 对比 **0 误罚**(尤其 05 京N541E6 不复活、02 不开错单);
- ④ 空间/IoU/identity 阈值全 11 扫描透明;
- ⑤ **Fix A(`_absorb`/transient_green)、#3 时序门(T=6.0)、b2 `_narrow_members`/`member_tracks_all` 语义全部不动**(都已 merge 受保护)。

## 7. 红线(标准多智能体约束)

独立 worktree/clone 不进他人活 worktree;禁 `git add -A`(scoped adds);qw 署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`;GT 只作诊断 oracle 不喂生产推理;禁 `select_gtfree` 生产;先诊断→出方案过 cc plan-gate→实现→cc 效果 gate;不 push/merge,球回 cc。

---
*派单:cc(arbiter)。承 [[b2-tracking-fragmentation-blindspot]]、`docs/handoff/2026-08-19-cc-plangate-09-prior-relocate.md`、优先级一页(`6124e19`)。qw 出只读诊断报告 → cc plan-gate。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
