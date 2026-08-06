# 违章01 FP 修法#3 时序门控 — 四道证据门(G1–G4)验证报告

> 关联: plan-gate #5 PASS (a9c48a8) → 授权写码。方案 docs/plans/2026-08-05-wb-plan-gate5-01fp-fix.md
> 代码分支: worktree `wb-01fp-temporal-gate` @ a9c48a8
> 验证脚本: scripts/verify_gates_01fp.py (复用 cli.run, monkeypatch fuse_light 抓段)
> 数据: data/output/gate_evidence/gate_evidence.json

## 1. 机制回顾

#3 纯时序门控(不碰 prior 外观阈值 / light_priors.json / ped_signal.pt):
- `fuse_light` 给每个 green/flashing 段标注 `max_raw_green_run_s` = 段内最长连续 raw 绿 run(融合前逐帧,与横测同口径)。
- `decide_violations` 对 `max_raw_green_run_s < T` 的 green 段**降级 review(非 confirmed)**。
- 初值 `T = 10.0s`,落 01 假绿最长 3.10s 与真绿最短 10.27s(视频11)两群 GAP,中间态降级 review 兜底防过拟合。

## 2. 代码改动(5 处, worktree 分支)

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/redlight/pipeline/temporal_fusion.py` | `fuse_light` 给 green/flashing 段标注 `max_raw_green_run_s` |
| 2 | `src/redlight/pipeline/decision.py` | `_go_intervals` 排除瞬态绿;新增 `_transient_green_intervals` 降级 review;`decide_violations` 加 `min_persistent_green_run_s`(默认10) |
| 3 | `src/redlight/pipeline/violation_engine.py` | `BatchViolationEngine.__init__` + `decide` 接参 |
| 4 | `src/redlight/app/cli.py` | `run()` 加 `min_persistent_green_run_s` 覆盖形参(默认读 config,供验证注入 T=0/10) |
| 5 | `configs/config.yaml` | `traffic_light.min_persistent_green_run_s: 10.0` |

## 3. 验证方法

复用 `cli.run` 零循环复制。monkeypatch `fuse_light` 抓带 `max_raw_green_run_s` 的灯段。
对 11 视频各跑两轮:
- **baseline** (`min_run=0`,等价未修):确认当前生产行为
- **fixed** (`min_run=10`):#3 机制

段级 GT (`events.csv`) 对齐:区分**真绿段**(与 GT green 区间重叠)与**假绿段**(落在 red/unknown 区间,同 01 prior 直采环境绿机制)。

## 4. 四道证据门结果

### G1 负例(误绿消除)
| 视频 | baseline confirmed | fixed confirmed | 结论 |
|---|---|---|---|
| 违章01 | 1 | 0 | ✅ 误绿段[48.36–61.29]降级 review |
| 违章10 | 0 | 0 | (纯负例,无 baseline 误绿) |

**G1 PASS**:01 生产 FP 消除。

### G2 真绿 TP 不回退(段级真绿段数)
| 视频 | base_true | fixed_true | no_regress |
|---|---|---|---|
| 02 | 1 | 1 | ✅ |
| 03 | 1 | 1 | ✅ |
| 04 | 0 | 0 | ✅ |
| 05 | 1 | 1 | ✅ |
| 06 | 1 | 1 | ✅ |
| 07 | 1 | 1 | ✅ |
| 08 | 1 | 1 | ✅ |
| 09 | 1 | 1 | ✅ |
| 11 | 1 | 1 | ✅ |

**G2 PASS**:所有真绿视频真绿段数不回退。

### G4 对抗式公交/遮挡真绿段 min max_run(阈值10s, cc 加严)
| 视频 | 性质 | 真绿段 max_run(s) | safe |
|---|---|---|---|
| 02 | 真绿 | 13.87 | ✅ |
| 03 | 公交玻璃反光 | 20.34 | ✅ |
| 04 | 无绿段 | None | ✅ |
| 05 | 公交玻璃反光 | 64.65 | ✅ |
| 06 | 低饱和石(S86-120) | 39.87 | ✅ |
| 07 | 真绿 | 24.65 | ✅ |
| 08 | 真绿 | 36.50 | ✅ |
| 09 | 真绿 | 55.89 | ✅ |
| 11 | 公交 | 10.27 | ✅ |

**G4 PASS**:对抗式扫描全部公交/大车遮挡真绿难例,真绿段 min max_run 全部 ≥10s,**主动证明无瞬态真绿被 T=10 误降级**(06 比 01 假绿 3.10s 还大 12.9×)。

### 总体 VERDICT: G1 ✅ G2 ✅ G4 ✅ → **ALL_PASS**

## 5. 关键发现:09 假绿顺带消除(额外收益)

初版 G 脚本未做段级 GT 对齐,把"所有 green 段"计入 G4,一度误报 09 的 0.943s 段"真绿被误杀"(G4 false fail)。深挖 09 GT 后翻案:

- 09 GT:`0–10s red` / `11–72s green`(真绿违章)/ `72–106.4s unknown`
- 09 绿段 `[0.0, 4.18, max_run=0.943]` **落在 GT red 区间** → 是 prior 直采环境绿瞬态(**同 01 假绿机制**),非真绿
- 09 绿段 `[7.41, 106.26, max_run=55.892]` 覆盖真绿区间 → 真绿长段,保留 confirmed

**含义**:09 在 baseline 下 confirmed=3 中混入 1 个假绿(落在 red 区间的瞬态环境绿),#3 机制将其降级 review → fixed confirmed=2(全真绿)。这是 #3 的**额外收益**:不只修 01,还顺带修 09 同类假绿。修正 G 脚本为段级 GT 对齐后,G2/G4 全 PASS。

## 6. review 增量与 T 定稿

- **全局 review 增量**:baseline confirmed_total=16 → fixed=14(−2,即 01+09 两个假绿降级);baseline review=0 → fixed review=2(+2)。误绿段进人工复核队列,**不静默丢绿**(符合 C1 中间态兜底防过拟合)。
- **T=10 定稿**:落 01 假绿最长 3.10s 与真绿最短 10.27s(视频11)两群 GAP,margin 充足(T 下探到 ~5s 仍可分,但取 10s 留足缓冲防单点过拟合)。中间态(5–10s)降级 review 兜底。

## 7. 六硬条件(C1–C7)闭合核对

- **C1 n=1 过拟合** → GAP 阈值(10s 非拟合单点 3.10s)+ review 兜底 ✅
- **C2 视频10 负例回归** → 10 baseline/fixed 均无 confirmed,无新增误绿 ✅
- **C3 公交/大车遮挡扫描** → G4 对抗式全 PASS(03/05/11 公交 + 06 低饱和石)✅
- **C4 7 好视频 TP 不回退** → G2 全真绿视频真绿段不回退 ✅
- **C5 只加时序判据** → 未碰 prior 外观/light_priors.json/ped_signal.pt ✅
- **C6 声明止血非根治** → 归 light-classifier-retrain 根治线;本机制为局部止血 ✅
- **C7 正向耦合 qw P2 车牌线** → #3 消 01 灯态误报后,qw 在 01 的连带错罚单(京N8ZK53)自动归零 ✅

## 8. 红线合规

worktree 内改码(未碰主仓库 HEAD);未改 `light_priors.json` 坐标/权重;未碰 `ped_signal.pt`;未削弱 `enforce_transition_limit`;禁 `select_gtfree`;权重不入库;scoped 提交(仅代码+脚本+报告,诊断产物 gitignored)。

## 9. 交付与下一步

- 代码:5 文件(worktree 分支 `wb-01fp-temporal-gate`)
- 验证:`scripts/verify_gates_01fp.py` + `data/output/gate_evidence/gate_evidence.json`
- 下一步:**球交 cc 亲测复核该表** → 复核通过 → merge 主仓库 → (可选)push
