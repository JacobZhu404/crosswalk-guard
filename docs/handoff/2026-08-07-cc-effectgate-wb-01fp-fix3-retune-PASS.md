# cc 最终效果 gate:wb #3 时序门控 retune T=6.0(51912e0)— **PASS(merge wb-01fp-temporal-gate → main)**

> 审核对象:branch `wb-01fp-temporal-gate` tip `51912e0`(retune commit `1bdb299` + 报告/方案口径统一 `51912e0`)
> 署名:cc(效果 gate/独立复核)— 承窄送回 `29aa717`([[wb-01fp-temporal-plangate5-passed]])、brief `791158e`
> 复核环境:cc 独立 detached worktree `cc-verify-gate6@51912e0`(symlink input_video/models),**全新 /tmp/cc_gate6_verify output,未复用 wb 任何 JSON**,`verify_gates_01fp.py --min-run 6.0` 从零跑 01/10/11/06/07。
> **结论:送回四条全部达成。cc 独立从零复现 G1/G2/G4 于 T=6.0:违章01 confirmed 1→0、违章11 真绿段 10.266s 距阈 +71%(送回时 T=10 仅 +2.7%,悬崖已离开)、真绿零回退、ALL_PASS=True。retune diff 机制零漂移(只改 5 处生产码默认值 + config 注释 + 验证脚本受参驱动)。config 假注释已修。口径统一全 11 视频。PASS,merge。**

---

## 1. cc 独立从零复现(T=6.0,不复用 wb JSON)

| 门 | cc 复现结果 | 判定 |
|---|---|---|
| **G1**(负例消误绿) | 违章01 baseline_confirmed=1 → fixed=0(绿段 [48.36,61.29] max_run **3.098s** <6.0 降级 review);违章10 0→0(无新 FP) | ✓ |
| **G2**(真绿不回退) | 违章06/07/11 全 no_regress=True(fixed_true_green_segs=baseline) | ✓ |
| **G4**(遮挡碎真绿对抗扫描) | 真绿 min max_run:06=31.22s / 07=24.645s / **违章11=10.266s**,全 ≥6.0 safe | ✓ |
| verdict | G1=True G2=True G4=True **ALL_PASS=True** | ✓ |

review_delta(本 5 子集):confirmed 4→3,review +1(即违章01 降级)。与 wb 全 11 报告 +2/16→14 自洽(全集多出的降级为 09 假绿,未在本子集)。

## 2. 决定性:违章11 悬崖已离开(送回核心诉求达成)

- 物理绿段值**未变**(01=3.098s / 11=10.266s / 06=31.22s / 07=24.645s),只挪阈值。
- 送回时 T=10:11 距阈 = 10.266/10.0 = **+2.7%**(悬崖边,检测抖动即丢 TP)。
- 现 T=6.0:11 距阈 = 10.266/6.0 = **+71%**(GAP 中部,健康 margin);01 距阈 = 3.098/6.0 = **−48%**(仍消除)。
- **单调性护栏**:T=6.0 严格落在「已观测假绿最长 3.10s」与「真绿最短 10.27s」之间。降 T(10→6)只会**减少**降级 → G2 只增不减(不可能新引真绿回退);且 6.0 > 全部已观测假绿(01=3.10 / 09=0.94)→ **不复活任何已观测假绿**。两侧安全。

## 3. retune diff 机制零漂移(cc 亲核 `1bdb299`)

- 改动仅:5 处生产码默认值 `10.0→6.0`(config.yaml:79 / cli.py:66 / decision.py:10,56 / violation_engine.py:139)+ config 注释修正 + 验证脚本 `--min-run` 默认 6.0 + fixed 轮/G4 safe 判据改受 `args.min_run` 驱动 + resume 升级。
- **机制未动**:`_transient_green_intervals`(瞬态绿→review 非硬杀,C1 兜底)、`_go_intervals` 过滤、`max_raw_green_run_s` 标注、engine/cli/config 接线 —— 全部保留。
- **验证脚本 resume 升级**(仅复用 min_run 匹配的 fixed,否则重跑)= 正防「陈旧 T=10 JSON 被误复用」的坑,且在验证脚本非生产码,不影响生产行为。良好工程。
- config 假注释已修:`26.8s(子集)` → `真绿最短10.27s@违章11` + 正确 margin(−48%/+71%)。生产码无残留 10.0(git grep 净)。
- 红线合规:未碰 `_sample_roi`/`sat_min`/`light_priors.json`/`ped_signal.pt`;禁 select_gtfree;GT 仅段级当裁判;生产 v2/box。

## 4. 送回四条核销

| # | 送回条件 | 状态 |
|---|---|---|
| ① | T 重调 GAP 中部两侧 ≥30% margin | ✓ T=6.0,01 −48% / 11 +71% |
| ② | 修 config 假注释 | ✓ 26.8s → 10.27s@违章11 |
| ③ | 新 T 重跑 G1/G2/G4 | ✓ cc 独立从零复现 ALL_PASS |
| ④ | 口径统一全 11 视频 | ✓ 主交付报告+方案已改;历史横测/cc handoff 作准确历史留存不篡改 |

## 5. 裁定与 merge

**PASS。** wb #3 时序门控(retune 后)cc 独立复现全部成立,机制零漂移,送回四条全达成。wb 明确将 merge 交回 cc(不 push/merge)→ **cc 执行 merge `wb-01fp-temporal-gate` → `main`**。

**merge 顺序(C7 正向耦合)**:wb #3 先行 → 消 01FP 事件 → qw P2/P1 车牌线在 01 上的连带错罚单(京N8ZK53/京ADF5307)自动归零。qw plate-fix(已 PASS `ebb025a`)随后 merge,cli.py 若有文本冲突由 qw merge 时解决(两者改 cli.py 不同函数)。

## 6. 一句话给 Jacob

wb 按我送回的要求把阈值从 10 秒降到 6 秒——我自己开独立工作区、用全新输出**从零重跑**(没碰 wb 那份跑过的数据),验出来:违章01 的误报照样消掉,违章11 那段真绿现在离阈值 **+71%**(之前卡在 +2.7% 的悬崖边),真绿一个没丢,config 假注释也改对了。机制一行没动,只挪了个数。**判 PASS,我这就把 #3 合进 main。** 合完违章01 那张灯态误报带出来的错罚单自动消失(qw 车牌线的连带错单也一起归零)。

---
*署名:cc(效果 gate/独立复核)。证据=cc detached worktree@51912e0 全新 output 从零跑 verify(违章11 fixed max_run 10.266s/margin +71%,01 confirmed 1→0,ALL_PASS=True)+ retune diff 机制零漂移核对。承 [[wb-01fp-temporal-plangate5-passed]]、[[crosswalk-ceiling-finding]](11 硬救回 TP)、[[qw-p2-plate-effectgate-fail]](C7 耦合)。*
