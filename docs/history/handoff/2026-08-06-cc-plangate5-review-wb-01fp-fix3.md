# cc plan-gate #5 审方案:wb 违章01 FP 修法 #3 时序门控(6124470)— **PASS(放行写码 + 交 G1–G4)**

> 审核对象:`docs/plans/2026-08-05-wb-plan-gate5-01fp-fix.md`(commit `6124470`)— #3 时序门控方案预审稿
> 署名:cc(plan-gate/独立复核)— 承 [[wb-01fp-temporal-plangate5-passed]](可分性 CONFIRMED 授权出方案)、[[prior-misframe-rootcause]]、[[light-classifier-retrain]]
> **结论:方案逐条应答我在 plan-gate #5(可分性 gate)钉的六硬条件(C1–C7),机制最小(两处:fuse_light 标注 + decide_violations 门控)、纯加性时序判据、中间态降级 review 非硬杀、阈值落两群 GAP 非拟合单点、证据门 G1–G4 结构正确、全程未写生产码。PASS,放行 wb 建 worktree 写码,按 G1–G4 交证据表,再走最终效果 gate。**

---

## 1. 六硬条件应答核验(cc 逐条)

| # | 条件 | wb 应答 | cc 核验 |
|---|---|---|---|
| C1 | n=1 过拟合(最重) | 阈值 T=10s 落 01(3.10s)与真绿(26.8s)两群 GAP;中间态(5–10s)降级 review 非硬杀;T 不预锁,G1/G2 后定稿 | **合格**——降级 review 是关键(最坏进人工队列不静默丢绿);"拟合两群间距非单点"论证成立 |
| C2 | 补负例 10 回归 | G1 前置证据门:跑视频10 确认 maxGreenRun 不与真绿重叠、01+10 均 0 FP | **合格**——列为放行前置 |
| C3 | 公交/大车遮挡碎真绿 | G4 专项扫 05–09/10「长绿段但最长 raw run<T」实例,确认是瞬态而非应 confirmed 的真短绿;有则 T 上修/回退讨论 | **合格但为本方案最大风险**,见 §2 |
| C4 | 7 好视频全 TP 不回退 | G2 全 11 端到端,07 基准,confirmed TP 不降 + review 量不失控 | **合格** |
| C5 | 只加时序判据 | 改动限 fuse_light + decision.py 两处;不碰 _sample_roi/sat_min/light_priors.json/ped_signal.pt | **合格**——遵 [[prior-misframe-rootcause]] 不碰 prior 坐标、红线不碰权重 |
| C6 | 声明止血非根治 | 明示 #3 只消 01 单点 FP,根治归判别器重训线 | **合格**——遵 [[light-classifier-retrain]] |
| C7 | 正向耦合 qw P2 车牌线 | #3 消 01FP 后 qw 连带错罚单归零,建议 #3 先行 | **合格**——与 cc 效果 gate 第二轮 carve-out 一致 |

## 2. 放行同时钉死的证据门加严(写码后必须过,不得软化)

1. **G4 必须是对抗式扫描,不是"确认没有"**:C3 是本方案唯一能翻车的地方——若某真绿信号被公交/大车周期性遮挡致其**最长**连续 raw 绿 run 跌破 T,门控会把真绿降级 → 潜在真绿 FN。wb 须**主动构造/搜寻**这类实例(逐视频报每个真绿段的 max_raw_green_run_s 分布),证明真绿段最长 run 全 ≥ T 有 margin;**不能只报"未发现"**。发现即触发 T 上修或机制回退,不得隐藏。
2. **降级 review 的量必须报**:G2 除 confirmed TP 不回退外,须报 review 桶增量(11 视频),证明门控没把大批真绿冲进人工队列(review 爆表 = 变相不可用)。
3. **加性/字段隔离**:fuse_light 新增 `max_raw_green_run_s` 须为**纯加性字段**,不改既有 segment 的 green/flashing/unknown 判定与既有下游消费;decision.py 门控只新增排除分支,不动既有 _go_intervals 语义。回归证之。
4. **T 定稿留痕**:T 最终值须附 G1(视频10)+ G2 全量的判定表,写清为何取该值、margin 多少;不接受"沿用 10s 因为初值是 10s"。

## 3. 裁定

**PASS。放行 wb 建 worktree、按方案 §5 改 fuse_light + decision.py 两处、跑 G1–G4 交证据表。** 最终 merge 前 cc 亲测复核 G1–G4(尤其 G4 对抗扫描 + review 量),走最终效果 gate。

**红线重申**:scoped git(禁 `git add -A`)、独立 worktree(遵 [[multi-agent-worktree-isolation]])、禁 select_gtfree、不碰 light_priors.json/ped_signal.pt/权重、诊断产物 gitignored。

## 4. 一句话给 Jacob

wb 的 #3 时序门控方案(把"最长连续 raw 绿 run < 10s"的绿段降级 review、不硬杀)答全了我钉的六条,机制最小、只加时序判据不碰灯参、中间态降级兜底防过拟合——**我批了,放 wb 写码**。写完必须交四道证据门,我盯死其中一道:公交/大车遮挡会不会把某个真绿信号灯的最长绿段也压到 10s 以下被误降级——wb 得主动去找这种情况证明没有,不能只说"没发现"。过了这道 + 全量 F1 不回退,我再亲测复核放 merge。

---
*署名:cc(plan-gate/独立复核)。承 [[wb-01fp-temporal-plangate5-passed]]、[[prior-misframe-rootcause]]、[[light-classifier-retrain]]。*
