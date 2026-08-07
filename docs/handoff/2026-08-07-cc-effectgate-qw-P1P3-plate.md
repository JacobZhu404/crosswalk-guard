# cc 效果 gate:qw 车牌 P1(多牌)+ P3(ROI 重试)(068bfea)— **PASS(授权 merge plate-fix → main)**

> 审核对象:`068bfea`(branch `plate-fix`)+ 报告 `docs/reports/2026-08-04-qw-plate-fix.md`(`01cbbbd`)
> 署名:cc(效果 gate/独立复核)— 承 [[qw-p2-plate-effectgate-fail]](P2 v7.1 PASS)、[[measurements-disagree-find-the-bug]]、[[b2-tracking-fragmentation-blindspot]]
> 复核环境:cc 独立 detached worktree `cc-verify-plate2@068bfea`(symlink input_video/models),生产 `configs/config.yaml`(**v2/box**)端到端亲跑 11 视频 + 单测/集成回归。
> **结论:命中 7/12、真事件误罚=0、5/12 逐字保留 + 08/09 二车找回——cc 生产 v2/box 独立复现全部成立。#1 对抗风险(P3 在 05 复活错牌重开送回伤口)证伪:P3 确在 05/02 开火但双双返空,05 仍空串。加性(consensus._recompute 只读 text/conf/ts,box=False)/回归(unit 264 + integration 5 全过)干净。PASS,授权 qw merge plate-fix → main。**

---

## 1. 决定性证据:生产 v2/box 逐事件全表(cc 亲跑复现,GT 仅当裁判)

| 视频 | hv | cc 复现回填牌 | GT violating | 判定 |
|---|---|---|---|---|
| 违章01 | **负** | 京ADF5307 | — | 误罚(负例FP,**灯态线** wb #3 carve-out) |
| 违章02 | 1 | `''` | 京LNE560 | 空(P3 开火返空,斜角极限) |
| 违章03 | 1 | 京ABV3428 | 京ABV3428 | **命中 1/1** |
| 违章05 | 1 | **`''`** | 京ADH9206 | **空 ✓ P3 开火返空,送回伤口未重开** |
| 违章06 | 1 | 京N2LE10 | 京N2LE10 | **命中 1/1** |
| 违章07 | 1 | 京Q5D2N8 | 京ACG0878,京EJQ505,京Q5D2N8 | 命中 1/3(天花板,见 §4) |
| 违章08 | 1 | 京ACD5358\|**京ACW6553** | 京ACD5358,京ACW6553 | **命中 2/2** |
| 违章09 | 1 | 京AC63971\|**京NNM526** | 京AC63971,京NNM526 | **命中 2/2** |
| 违章11 | 1 | 京AFW1222\|京AC81321 | (GT 盲区) | 盲区(不可核验,记账②) |

04/10 无 confirmed 事件。**命中 = 7/12;真事件(hv=1 且有 GT 牌)误罚 = 0;harness 汇总「误罚=3」= 01负例FP(1)+ 11盲区两牌(2),全为 carve-out。**

## 2. 三项关键核验(cc 独立)

- **加性(5/12 保留)**:v7.1 五命中(03/06/07/08/09 主牌)在 068bfea **逐字不变**——尽管 P1「一车一牌」改了 agg 构造(每 tid 只贡献其最佳 text,非 v7.1 全 text 汇池),主牌选择对这 5 个未变。08/09 各 +1 二车 = 7/12。**加性经端到端坐实。**
- **#1 对抗风险(P3 重开 05 伤口)= 证伪**:P3 ROI 重试仅在主牌空时触发 → 在 confirmed 事件里只有 02/05 命中此条件(04/10 无 confirmed)。cc 复现下 **02/05 P3 均返空**,05 仍空串——[[qw-p2-plate-effectgate-fail]] 送回的 05 京N541E6 误罚**未复活**。斜角物理极限的归因(HyperLPR3 弱项)与「零收获」结果自洽。
- **加性(投票不污染)**:`plate_consensus._recompute` cc 亲验只读 text/conf/ts(`box in src == False`);consensus.py diff 仅一行 docstring;P1/P3 全部逻辑在 cli.py 下游对 agg 过滤,不碰 ED 投票。

## 3. 回归(cc 亲验)

- `tests/unit` = **264 passed**;`tests/integration` = **5 passed**(清 `__pycache__` 同名冲突后)。
- 唯一环境噪声:全量收集时 `test_signal_state_classifier.py` 在 tests/ 与 tests/unit/ 同名 basename → pytest 采集冲突(环境,非代码);wb 挖矿线 `test_mine_classifier_retrain`(tests/ 根,068bfea 未触碰)。**均非 qw 回归。**
- 红线合规:未碰 `_sample_roi`/`sat_min`/`light_priors.json`/`ped_signal.pt`;禁 select_gtfree;GT 仅段级当裁判不进推理;生产 v2/box。

## 4. 诚实归因采信(9/12 未达,非硬凑)

- **07 京ACG0878**(3 帧)与 03 京ABV200(3 帧非 GT)在「低帧 stationary 车组牌」不可区分——降全局门槛即 03 误罚。真实 tradeoff,信号天花板。**采信。**
- 07 京EJQ505 从未读到 / 02 京LNE560 / 05 京ADH9206 斜角 60° P3 零收获 = 识别层物理极限。→ **seg 检测模型独立立项交 Jacob**,采信不硬凑。

## 5. 记账(不阻断,交付物须带)

1. **11 盲区被 P1 从 1 牌扩到 2 牌**(京AFW1222+京AC81321)。GT 无牌 → 不可核验;若 京AC81321 是幻影则是**检测不到的错罚单**。qw 已标注记账②,cc 同意:**生产罚单此条须人工复核**。P1 多牌在盲区上放大了不可核验暴露面——诚实项,非缺陷。
2. **`_episode_plates` 定义两次**(068bfea 内完全相同,第二个覆盖第一个)= 冗余死码,下轮删。
3. **P1(`_pick_plates` 多牌)/ P3(`_p3_roi_retry`)无单测**——仅端到端验,无 unit 级覆盖(现有单测只覆盖 P2 三约束)。下轮补 P1 二车/P3 返空的单测。
4. **b2 过合并(17 tid 脏袋)独立立项**([[b2-tracking-fragmentation-blindspot]]):P1/P2 用「stationary 车组+质心聚集+代表 track 排除」绕开,不修根因,交 Jacob 排期。

## 6. 裁定与 merge 顺序

**PASS。授权 qw merge `plate-fix` → `main`**(scoped、署名、trunk=main)。plate-fix 不依赖 wb #3,可独立 merge;01 的 京ADF5307 错罚单在 wb #3 落地消 01FP 后自动归零(正向耦合,与 [[wb-01fp-temporal-plangate5-passed]] 一致)。**车牌线 P1/P2/P3 收官。**

## 7. 一句话给 Jacob

qw 的车牌 P1+P3 我亲跑生产配置 11 视频复现:**命中从 5 涨到 7(08/09 两辆违章车的牌都对了),真违章事件一张错罚单都没有**。我盯死的最大风险是 P3 那个「读不到就放大图重读」的机制会不会在违章05 上又读出上次送回的那张别车牌——**验了,05 上它确实开火了但读出来是空的,伤口没重开**。加性和回归都干净。**判 PASS,qw 可以把 plate-fix 合回 main。** 唯一提醒:违章11 GT 没标车牌,P1 给它回填了两张牌都没法核验,生产上这张罚单要人工复核(qw 已标注)。违章01 那张错罚单是灯态误报带出来的,wb #3 落地后自动消失。车牌线收官。

---
*署名:cc(效果 gate/独立复核)。证据=cc detached worktree@068bfea 生产 v2/box 端到端逐事件表(命中 7/12、真事件误罚 0、05/02 P3 返空)+ unit 264/integration 5 回归 + `_recompute` 读码证加性。承 [[qw-p2-plate-effectgate-fail]]、[[b2-tracking-fragmentation-blindspot]]、[[wb-01fp-temporal-plangate5-passed]]、[[01fp-falsegreen-fix1-disproved]]。*
