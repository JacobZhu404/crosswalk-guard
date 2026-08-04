# 斑马线 v2 接线报告(qw, C3 实施 + cc 三护栏验收)

> Jacob 拍板接线(C3): v2 时序聚合 + denom=box + box_overlap=0.20 接 `cli.run` 默认, 推翻 D2(mask 分母)语义。
> cc 已批准接线方案(`docs/plans/2026-08-04-qw-crosswalk-wiring-plan.md` + gate `docs/handoff/2026-08-04-cc-gate-qw-crosswalk-wiring-plan.md`), 4 项落定修正全部并入实施。
> qw 在隔离 worktree(`crosswalk-guard-wiring`, 分支 `wiring-b1`)实施, cc 逐项独立验收。

## 0. 接线 diff(最小面, 纯加性)

| 文件 | 改动 |
|---|---|
| `configs/config.yaml` | `crosswalk` 段加 `version: "v2"`(词表 v11/v2, 与 eval_violations --detector 对齐) + `occ_denom: "box"`; 注释声明与 method 正交; 回退=两字段改回 v11+mask |
| `src/redlight/app/cli.py` | 默认 detector 按 `cfg.crosswalk.version` 选 v2/v11; 默认 occ_denom 取 `cfg.crosswalk.occ_denom`; **显式传参仍优先**(诊断/评测注入路径不变) |
| `scripts/eval_tracking.py` | 显式钉死 `CrosswalkDetector(cfg)` + `occ_denom="mask"`(保 b2 跟踪评测口径, cc gate 修正①) |
| `scripts/diag_vehicle_track_fragmentation.py` | 同上显式钉死 v11+mask(保 b2 碎片化 baseline) |
| 引擎/检测器/dag/visualizer | **零改动**(box_overlap 早已落地, v2 detect 签名与 v11 兼容, dag 生产节奏天然匹配 running-max) |

**影响面**(cc 全仓枚举 + qw 复核): 显式传参调用方(eval_violations/sweep/generate_report/diag_*)输出不变; 受影响仅不传参入口(run_video.py / cli main)——即接线意图。cc 抓到的 2 个漏网 no-param 调用方(eval_tracking:34 / diag_vehicle_track_fragmentation:174)已显式钉死。

## 1. 护栏①: 不接线默认路径 bit-identical —— PASS

协议(cc gate 修正③, 先立确定性地板再归因):
1. **接线前(HEAD e72f336)连跑两次** 不传参 `cli.run`(= v11+mask 现状默认) 全 11 视频 → `data/output/qw/wiring_pre/{run1,run2}/`(快照工具 `scripts/qw_wiring_snapshot.py`, 禁标注视频/截图, 仅 violations.csv + sha256)
2. **接线后** config `version="v11"`+`occ_denom="mask"` 再跑同 11 视频 → `wiring_post/post/`
3. diff: 确定性地板 ∧ 回退路径均须 0 diff

| 步骤 | 结果 |
|---|---|
| run1 vs run2(确定性地板) | **0 diff, 11/11 视频** |
| post(回退 v11+mask) vs run1 | **0 diff, 11/11 视频** |
| 逐视频事件(run1, 与 cc 亲跑基线吻合) | 01:1 02:2 03:1 04:0 05:3 06:1 07:3 08:1 09:2 10:0 11:0(即 cc 773dff4 §0(B) 基线 TP=7/FP=7/FN=2) |

**结论**: 接线是纯加性的——diff 完全归因于接线本身; 回退开关(config 两字段)行为与接线前逐字节一致。

## 2. 护栏②: 2×2 消融(拆 v2 时序 vs denom=box 贡献) —— PASS

harness: `cli.run` + `match_violation_events(min_overlap_s=0.5)`, 同 eval_violations.py 口径
(`scripts/qw_wiring_ablation.py` 聚合)。

| 组合 | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| v11+mask(基线, 复用护栏①快照) | 7 | 7 | 2 | 0.500 | 0.778 | **0.609** |
| v2+mask(新跑) | 8 | 1 | 1 | 0.889 | 0.889 | **0.889** |
| v11+box(新跑) | 7 | 11 | 2 | 0.389 | 0.778 | **0.519** |
| v2+box(0.20, 复用 sweep 55 份) | 8 | 1 | 1 | 0.889 | 0.889 | **0.889** |

**贡献拆解(关键发现)**:
- **v2 时序聚合 = +46% 的全部来源**: v11+mask → v2+mask, ΔF1 **+0.280**(清 6 碎片 FP 7→1 + 救回违章11)
- **denom=box 单独是负收益(负对照)**: v11+mask → v11+box, ΔF1 **-0.090**(FP 7→11: 违章02 +1/08 +2/09 +4)——v11 全宽水平带 mask 在 box 分母下"任何停在带内的车都判压线"。**证明 box 分母必须搭配 v2 窄梯形 mask, 不能搭全宽带**
- **v2 下 box 与 mask 事件级等价**: v2+mask == v2+box(8/1/1 全同)——v2 窄梯形 mask 已把重叠分布推到阈值两端(与 sweep 5 档 box_overlap 全一致的现象一致), denom 事件级中性; **box 保留的理由是语义更合理(车足迹压线) + 与 Visualizer 标注一致, 非事件级增益**
- 联合: 0.609 → 0.889(+46%), 与 cc 亲跑同口径一致

## 3. 护栏③: 相机运动样本 + running-max 漂移泛化 —— 探针完成(负结果, 有发现)

- **真实样本现状**: 现有 11 视频全固定机位监控视角, **无相机大幅运动样本** → 真实运动样本记 **Phase 2 独立立项交 Jacob**; 本接线**不宣称已泛化到运动镜头**。
- **合成漂移探针**(`scripts/qw_drift_probe.py`, detector 层直测, 生产节奏每 4 采样帧调一次 v2.detect; 代表视频 02/06/08): 四档
  - `static`(原帧直喂, 基线) / `jitter`(每帧 ±1.5%平移±0.5°±1%缩放, 净位移≈0)
  - `pan`(每采样帧 0.5px 单向平移, 累计 ~25% 帧宽, 接近真实监控微漂移上限)
  - `pan_fast`(每采样帧 2px, 累计 ~整幅扫过, cc 修正④要求的**必触发拖影的失效模式档**)
  - 指标: mask 面积时间序列(拖影=单调膨胀)、末/首面积比、质心漂移

| 视频 | 档位 | 首面积 | 末面积 | 峰值 | 拖影(末/首) | 质心漂移px |
|---|---|---|---|---|---|---|
| 违章02 | static | 135195 | 539474 | 539503 | 3.99 | 192.6 |
| 违章02 | jitter | 128398 | 539914 | 539939 | 4.21 | 159.3 |
| 违章02 | pan(0.5px/帧) | 135195 | 540407 | 540653 | 4.00 | 188.3 |
| 违章02 | pan_fast(2px/帧) | 135195 | 486541 | 491357 | 3.60 | 154.9 |
| 违章06 | static | 635013 | 1206946 | 1206946 | 1.90 | 73.5 |
| 违章06 | pan_fast | 635013 | 1173446 | 1174637 | 1.85 | 76.3 |
| 违章08 | static | 701338 | 1205374 | 1205732 | 1.72 | 109.0 |
| 违章08 | pan_fast | 701338 | 1175523 | 1175945 | 1.68 | 130.5 |

(全档 jitter/pan 与 static 同量级, 末面积/static 末面积 = 0.97~1.00; 完整表见 `data/output/qw/drift_probe_result.txt`)

**判定与发现**:
- **未观测到拖影膨胀(负结果, 与 cc 预期失效模式相反但有利)**: pan_fast(累计整幅平移) 的拖影系数(3.60/1.85/1.68)与 static(3.99/1.90/1.72)同量级, 末面积不增反略减。机制: `_fit_trapezoid` 的**行 5-95 百分位裁剪**把运动弥散(宽而浅的拖尾)结构性裁掉, 加上运动帧的全局亮度/边缘响应稀释, running-max 的拖影被双重抑制。
- **对称抖动档** ≈ static(净位移≈0 不产生拖影, 符合预期)。
- **局限(如实记录)**: 合成仿射 ≠ 真实镜头运动(透视畸变/新场景入画/遮挡/光照变化); mask 面积是代理指标非端到端事件级。**真实运动镜头泛化未验证, Phase 2 必须用真实样本补测**(独立立项交 Jacob)。固定机位生产(11 视频全监控视角)不触发拖影, 接线安全。

## 4. 端到端复现(接线后默认路径) —— PASS

接线后默认 config(version=v2 + occ_denom=box) 不传参跑全 11 视频(`scripts/qw_wiring_snapshot.py --tag e2e`, 禁标注视频) → `match_violation_events` 聚合:

| 指标 | 实测 | 预期(cc 773dff4 重建) |
|---|---|---|
| TP/FP/FN | 8/1/1 | 8/1/1 |
| P/R/F1 | 0.889/0.889/**0.889** | 0.889/0.889/**0.889** |

逐视频与 cc 重建完全吻合: 02/03/05/06/07/08/09/11 TP, 违章01 FP(既有, [48.4-61.3] 窗口), 违章04 FN(灯态), 违章10 干净。

**性能**(冒烟实测, 违章02): v11+mask 61.8s vs v2+box 74.3s ≈ 1.2×, CPU 上可接受(v2 每 detect 全帧 Sobel×1 + running-max, 与 v11 多条带扫描同量级)。

## 5. 结论与后续

- **接线完成且加性成立**: 护栏① 11/11 bit-identical(接线前双跑 0-diff 确定性地板 + 回退路径逐字节一致); 生产默认 = v2+box; 回退=config 两字段改回 v11+mask
- **消融定论(护栏②)**: v2 时序聚合 = +46% 的全部来源(0.609→0.889); denom=box 在 v2 下事件级中性(语义更合理 + 与可视化一致), 单独(v11+box)负收益(FP 7→11)——**box 分母必须搭配 v2 窄梯形 mask**
- **漂移探针(护栏③)**: 四档合成漂移(含单向 pan/pan_fast)未观测到拖影膨胀(_fit_trapezoid 行百分位裁剪结构性抑制); 但真实运动镜头泛化未验证 → **Phase 2 独立立项交 Jacob**(采集运动镜头 + GT 标注 + 正式评估)
- 违章01 FP / 违章04 FN 为既有老问题(非接线引入, cc C4 已证)
- **待 cc 逐项独立验收**(护栏① 0-diff / ② 2×2 / ③ pan 探针 / 端到端 0.889 复现); 通过后 wiring-b1 merge 回 main; 接线 commit `f792856`(wiring-b1 分支, 待 merge)

## 方法学

- 快照/消融 harness: `cli.run`(preset=balanced) + `match_violation_events(min_overlap_s=0.5)`, 与 eval_violations.py 同口径
- 确定性: 快照工具 multiprocessing fork, jobs=4; 接线前双跑 0-diff 证明同环境确定性(跨 jobs/环境差异不承诺)
- 数据产物: `data/output/qw/wiring_pre|post|abl_*|wiring_e2e|drift_probe_result.txt|sweep_*`(gitignored, 供 cc bit-for-bit 复核)
- 工具: `scripts/qw_wiring_snapshot.py`(①快照) / `qw_wiring_ablation.py`(②聚合) / `qw_drift_probe.py`(③探针)
