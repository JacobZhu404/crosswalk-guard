# cc plan-gate:qw b2 脏袋过合并诊断 = 诊断 PASS / 决策上交 Jacob(评测语义 + P 安全修法)

> 署名:cc(arbiter) 转交:Jacob(评测语义拍板 + 修法放行);抄送 qw/wb/ds
> gate 对象:分支 `qw-dirtybag-diag`@`1ab8a59`(worktree `crosswalk-guard-qw-dirtybag`,未 push;报告 `docs/reports/2026-08-19-qw-dirtybag-diagnostic.md` + `scripts/qw_dirtybag_diag.py`)
> cc 独立复现:**load-bearing 断言在 main 上独立验证**(不依赖 qw 未 push 的 worktree)——读生产 eval 码 + events.csv + 用生产 `match_violation_events`/`classify_false_positives`/`aggregate` 跑端到端仿真。

---

## 1. cc 独立复现的核心断言(全部成立)

**① 评测契约锁 = 真的(端到端仿真,生产 eval 函数):**
| 场景 | tp | 主 P | p_only_true_fp | frag |
|---|---|---|---|---|
| A. b2 现状(单 episode 覆盖窗) | 1 | **1.0** | 1.0 | 0 |
| B. 拆成 锚+真违章车(京ACG0878,都在窗内) | 1 | **0.5** | 1.0 | 1 |
| C. φ 窗收窄(仍单 episode) | 1 | **1.0** | — | 0 |

→ **拆开(哪怕拆出 GT 表内真违章车)→ 主 P 1.0→0.5**;被拆出的 pred 被 `classify_false_positives` 判 `fragment`(与 GT 窗时间重叠≥0.5s),故 `p_only_true_fp` 不动、主 P 崩。**qw"拆真车=涨宽度在当前测评不成立(破 P)"是测量事实,cc 用生产 eval 码坐实。**

**② 1:1 前提成立:** events.csv 每正例视频恰好 1 行 `is_violation=1`(02/03/04/05/06/07/08/09/11=9 窗,01/10 负例)。`match_violation_events` 贪心 1:1 → 每窗只认 1 pred,其余全进 raw `fp`(:190)→ 主 `precision`(:192)。

**③ 过合并存在性:** 已是 **cc 自己 2026-08-03 的独立铁证**(07 track20 与锚整窗共存 IoU 0.122<0.3),qw 本次 re-measure(median IoU 0.0/cd 960px)方向一致=确证,非新事实。

## 2. cc 判定:这是**评测语义决策**,不是编码选择(qw 定性正确)

b2 效果 gate 的硬条件①"P=1.000"= **主 precision**(=b2 现状的 P)。在主-P 契约下:
- **任何同窗拆分(α/β)→ +1 raw fp → 破 P=1.000**,与拆出的车是不是真违章无关。
- **只有 φ(窗收窄、不拆、仍单 episode)P 安全。**

但存在第二个 precision(`p_only_true_fp`,:244,排除 fragment):同窗拆出真违章车对它**零影响**。**这正是要 Jacob 拍的评测语义**:同窗内把真违章车拆成独立记分,算不算 precision 的罪?
- 若判**算**(现状=窗级/每过街窗 1 票)→ 脏袋**真拆不了**,只能 φ 收窗 + 车牌侧治理。
- 若判**不算**(车级/每违章车 1 票,即认 `p_only_true_fp` 为准)→ α 拆分合法,脏袋可真正解合并,但这是**评测契约级改动**,需独立立项 + tracking-GT null 框。

## 3. cc 追加发现(比 qw α/β/φ 分类更尖锐,直指头号伤 02 错罚)

**φ(时间窗收窄)修不了 02 错罚。** qw 自证 02 铁证搭车 `京A12222`(tid68)的 2 个 confirmed 片段**在 GT 窗内** → 时间窗收窄只剔窗外成员,**剔不掉窗内搭车** → 02 错罚照旧。

**但有一条 qw 分类漏掉的 P 安全修法:身份过滤 `member_tracks_all`(车牌回填源)、不拆 episode。** b2 gate 已坐实 `member_tracks_all` **只被车牌回填消费、无记分消费者**。∴ 把空间相离的窗内搭车(京A12222)从 `member_tracks_all` 剔除 → 车牌回填不再抓它 → **修 02 错罚 + 零记分影响 + 不破 P + 不依赖评测语义拍板**。这与 φ(时间)正交:**φ 治窗外碎片(记分指标),身份过滤治窗内搭车(车牌错罚)。头号伤要的是后者。**

*(caveat:回填的选牌逻辑 P1 多牌/P2 span 隔离/P3 ROI 重试较绕,身份过滤是否干净须在方案里对 05 京N541E6 不复活 + 02 不开错单逐一验;不是拍脑袋结论。)*

## 4. 诚实记账:cc 复现深度

- **load-bearing(评测锁 + 1:1 前提 + 过合并存在)= cc 独立 bit-for-bit 坐实**(生产 eval 仿真 A/B/C + events.csv + 自己 2026-08-03 铁证)。
- **逐视频 A/B/U 整数计数(B 类误并数、盲盒 97、拆开新增 123 等)= 未逐数重跑**(qw 分支未 push,relay 表格错位不可据)。这些量化脏袋规模,但**不改修法方向裁定**(方向由评测契约推出,与 B=64 还是 60 无关)。若 Jacob 要把计数也纳入 bit-for-bit gate:qw push `qw-dirtybag-diag`,cc 在自建 gate worktree 跑 `qw_dirtybag_diag.py` 复现(同 b2 流程)。qw 提议的"11 窗拆开仿真表"cc 不需要——§1 端到端仿真已自建决定性证据。

## 5. 裁定 + 转 Jacob 两决策

**诊断 PASS**:评测契约锁独立坐实,定性正确(拆不动=测量事实非观点),球回口径干净,红线合规(只读/未 push/未碰生产码)。

**D1(立即、低风险、不依赖评测拍板)—— 建议放行 qw 出方案:**
P 安全治理包 = **φ 窗收窄(剔窗外成员,治记分脏袋指标)+ 身份过滤 `member_tracks_all`(剔窗内空间相离搭车,治 02 错罚)**,两者都不拆 episode、by-construction 不破主 P。走 plan-gate(硬条件承 b2 五条:P=1.000/逐 episode 审计/车牌 0 误罚含 05 不复活 02 不开错单/阈值透明/Fix A·#3·b2 语义不动)。

**D2(战略、评测契约级)—— 需 Jacob 产品意图拍板:**
产品是"每过街窗 1 张罚单"(窗级=现状主 P)还是"每违章车 1 张罚单"(车级=`p_only_true_fp`)?**只有判车级,脏袋才谈得上真解合并(α)**,且要 Jacob 补 tracking-GT null 框(02/07/09 各 2-3 帧 anchor)才能判 97 盲盒身份。判窗级则 α/β 永久出局,脏袋治理止于 D1。

**未决(qw §9 球回,采信):** ①tracking-GT null 框缺失→97 盲盒不可判身份;②评测语义 Q1-Q3;③11 GT 无牌。

## 6. 红线合规
cc 未进 qw 活 worktree(全部验证在 main + 生产 eval 仿真);未 push/merge qw 分支;零生产码改动;GT 仅 oracle。

---
*署名:cc(arbiter)。证据=生产 eval 端到端仿真(拆真违章车→主 P 1.0→0.5、φ→1.0)+ events.csv 9 窗 1:1 坐实 + 2026-08-03 自有过合并铁证。承 `docs/handoff/2026-08-19-cc-brief-qw-b2-dirtybag-rootcause.md`、[[b2-tracking-fragmentation-blindspot]]、[[eval-methodology-gap-overfit]]、[[plate-line-diagnosis]]。*

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
