# cc → qw:D1 放行(P 安全治理包)出方案 — Jacob 拍板"继续"

> 署名:cc(arbiter,代 Jacob 放行 D1) 抄送:Jacob/wb/ds
> 承:`docs/handoff/2026-08-19-cc-plangate-qw-dirtybag-diag.md`(`f21cef8`,诊断 PASS)。Jacob"继续"= 放行 D1 出方案;**D2(评测语义,窗级 vs 车级)仍挂起待 Jacob 产品意图,D1 设计成不依赖 D2、不拆 episode。**

---

## 1. D1 scope(出方案,先诊断后写码,走 plan-gate)

**头号交付 = 身份过滤 `member_tracks_all`(车牌回填源)**:把与锚**空间相离**的窗内搭车(如 02 `京A12222` tid68)从回填成员集剔除 → 断 02 错罚根因。**不拆 episode、b2 已证 `member_tracks_all` 无记分消费者 → by-construction 不破主 P。**

**次要交付 = φ 窗收窄**:剔窗外成员,清碎片化指标。**注意其价值有限**(member_tracks 无记分消费者=对 F1 零改动如 b2;窗外成员不是 02 错罚源)——列为可选,别喧宾夺主。

## 2. ⚠️ 诚实定性(方案必须量化,供 Jacob 决 merge)

b2 效果 gate 已坐实**当前车牌输出已经 0 误罚**(05 空 / 02 被 P2 止血)。∴ **D1 大概率是"根因加固 + 指标卫生",不是修复当前活 bug**:
- 价值 = 拆掉 02 潜伏地雷(qw §3"未来车牌行改动=多牌/逐车回填有再败风险")+ 碎片指标更准。
- **方案里必须量化"当前 vs D1 后"的车牌逐 episode 逐字节 delta**。若与 b2 同为**零输出改动**,请如实说——Jacob 据此判"拆潜伏风险是否值得进 main"(可能像 b2 一样批准为加固基建,也可能判低优先级暂缓)。别把加固包装成 bug 修复。

## 3. 方案必答的设计问题(plan-gate 会卡)

1. **"空间相离搭车"判据(生产可用,非 GT)**:用 tracker 自身几何(成员轨迹 vs 锚的质心距/IoU 阈值),**不得用 GT 窗/GT 框**(生产推理无 GT)。97 盲盒是**诊断标注**限制(qw 无 GT null 框判 A/B/U),不是生产过滤限制——生产过滤跑 tracker 几何即可;但**gate 置信**(有没有误剔真碎片)在盲盒上受限,方案要讲清过滤在"能几何分辨的成员"上动手、盲盒成员保守保留(不剔)。
2. **φ 窗收窄的"窗"在生产怎么定义**:必须用 episode 自身的**违章证据窗**(绿+占道累积 span),**不是 GT 窗**——否则 GT 泄漏进生产=红线。这是从诊断(用 GT 窗)翻译到生产的关键,写错就废。
3. **不破已得车牌命中**:身份过滤绝不能碰掉 postmerge 已收的**真命中 5/12→7/12**([[postmerge-f1-875-09-regression]]、[[qw-P2-plate-effectgate-fail]]:P1 多牌/P2 span 隔离/P3 ROI 重试)。方案对 05 京N541E6 不复活 + 02 不开错单 + 08/09 二车真命中保留**逐一验**,不是拍脑袋。

## 4. plan-gate 硬条件(承 b2 五条,提前告知)

①主 P=1.000 零新增记分 FP ②逐 episode 审计全 11 ③**车牌全 11 与 BASE 对比 0 误罚**(05 京N541E6 不复活 / 02 不开错单 / 真命中不掉)④过滤阈值(质心距/IoU)全 11 扫描透明 ⑤**Fix A(`_absorb`/transient_green)、#3 时序门(T=6.0)、b2 `_narrow_members`/`member_tracks_all` 语义全部不动**。

## 5. 红线 + 流程

独立 worktree/clone 不进他人活 worktree;禁 `git add -A`(scoped);qw 署名 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`;GT 只作诊断 oracle 禁喂生产;先出方案 → cc plan-gate → 实现 → cc 效果 gate;不 push/merge 球回 cc。**先出方案,别直接写生产码。**

## 6. D2 挂起(qw 知悉,勿越界)

D2(评测契约窗级 vs 车级)是 Jacob 产品决策,未拍。**D1 方案不得含任何拆 episode 的改动**(那是 D2 地盘,现破主 P)。若诊断中发现"不拆就治不了某伤",如实记入方案 §未决、球回 Jacob,别自行跨到 D2。

---
*派单:cc(arbiter,代 Jacob 放行 D1)。承 `docs/handoff/2026-08-19-cc-plangate-qw-dirtybag-diag.md`、[[b2-tracking-fragmentation-blindspot]]、[[plate-line-diagnosis]]、[[postmerge-f1-875-09-regression]]。qw 出方案 → cc plan-gate。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
