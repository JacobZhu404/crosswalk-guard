# CC 方案关(plan-gate)审 qw 斑马线 v2 接线方案(d38ef43)— 有条件批准: 摸底逐点复核准确, 但影响面漏两个 no-param 调用方; 3 项必改 + 1 项摸底更正后放手实施

> 出自 cc(arbiter)。Jacob 已拍板接线(v2 时序聚合 + denom=box 接 `cli.run` 默认)。qw 交接线方案(d38ef43, 仅摸底+计划未实施, 守约)。**cc 不采信摸底声明, 逐点亲读对拍**: 接线点全部准确; 加性原理成立; **但"影响面仅 run_video.py / cli main"不完整——另有两个 no-param `cli.run` 调用方会静默翻到 v2+box**。裁定: **有条件批准**, 补 1 项摸底更正 + 3 项护栏/命名必改后, qw 隔离 worktree 实施, cc 验收。

## 0. 摸底逐点复核(cc 亲读, bit-for-bit)

| qw 摸底声明 | cc 核实 | 结论 |
|---|---|---|
| `cli.py:75` 默认 v11 detector | `src/redlight/app/cli.py:75` `crosswalk_detector if ... else CrosswalkDetector(cfg)` | ✅ |
| `cli.py:71-72` occ_denom 仅显式进 engine, 否则引擎默认 mask | :71-72 `if occ_denom is not None: _engine_kwargs["occ_denom"]=occ_denom`; engine `__init__` 默认 `occ_denom="mask"`(violation_engine.py:138) | ✅ |
| `cli.py:80` Visualizer `occ_denom or "mask"` | :80 属实 | ✅ |
| v11/v2 `detect(frame, vehicle_boxes=None)` 签名一致, 均返 uint8 (h,w) | crosswalk.py:37 / crosswalk_v2.py:39 签名逐字一致, 均 return uint8 mask | ✅ 接口天然兼容 |
| 引擎已按 `occ_denom=="box"` 选 box_overlap | violation_engine.py:144 `overlap_thr = p["box_overlap"] if occ_denom=="box" else p["overlap"]`; accumulate :180 `denom=self.occ_denom` | ✅ |
| dag 每 4 采样帧调一次 detect(生产节奏) | dag.py:80-84 `if ctx["proc"] % cw_int == 0: ctx["mask"]=cw.detect(frame, vb)`, cw_int=4; mask 在 ctx 持续, accumulate 每帧消费 | ✅ v2 running-max 生产节奏坐实 |
| visualizer denom/阈值与引擎同源, 零改动 | visualizer.py:20-21 `denom=occ_denom...; overlap=p["box_overlap"] if denom=="box"` | ✅ |
| v2 `_accum` 每视频新建无跨视频污染 | crosswalk_v2.py:26-27 `_accum=None` 于 `__init__`; cli.run 每视频新建 comp | ✅ |
| eval_violations 显式 v11+mask, 参数优先 | eval_violations.py:53 默认 `detector_name="v11", occ_denom="mask"`; :66/:73 显式传 `crosswalk_detector=det`+engine_kwargs | ✅ 不受影响 |

**摸底质量: 接线点与接口分析准确, 加性原理正确。唯一疏漏在影响面枚举(见 §1)。**

## 1. cc 独立发现: 影响面比摸底大——两个 no-param 调用方会静默翻 v2+box

cc 全仓枚举 `cli.run`/`run(cfg` 调用方(非 .venv), 逐个查是否传 `crosswalk_detector`/`occ_denom`:

| 调用方 | 传 detector/occ_denom? | 接线后 |
|---|---|---|
| `run_video.py:31` | ❌ 不传 | **翻 v2+box**(生产入口, 意图内 ✅) |
| `cli.py:260` main() | ❌ 不传 | **翻 v2+box**(生产 CLI, 意图内 ✅) |
| `generate_report.py:198` | ✅ `crosswalk_detector=vd, occ_denom="box"` | 不受影响 |
| `eval_violations.py:73` / `sweep_box_overlap.py:49` | ✅ 显式 | 不受影响 |
| `eval_tracking_ab_split.py:316` / `eval_tracking_gtfree.py:138` / `diag_gt_crosswalk_ceiling.py:81` | ✅ 显式 | 不受影响 |
| **`eval_tracking.py:34-35`** | **❌ 不传** | **翻 v2+box**(摸底漏) |
| **`diag_vehicle_track_fragmentation.py:174-177`** | **❌ 不传** | **翻 v2+box**(摸底漏) |

- **`diag_vehicle_track_fragmentation.py`**: :180 `confirmed=[ev ... status=="confirmed"]`, Q1「违章车被切几 ID」取 confirmed 事件的 member_tracks → **v2 改变哪些事件 confirm(清 6 碎片 FP/救回违章11)→ 碎片化数字直接漂移**。这是 [[b2-tracking-fragmentation-blindspot]] 记录的 baseline 所依赖的工具, baseline 在 v11+mask 下测得, 静默翻 detector = 度量口径被偷换。
- **`eval_tracking.py`**: 核心 frag 指标用 GT 锚框+track_samples 的 box IoU(与 crosswalk mask 基本无关, 影响小), 但仍会在 v2+box 下重跑整条管线、events 变化。

**这违反"度量必须可复现/同口径"红线([[measurements-disagree-find-the-bug]] 精神)**: 一个测跟踪碎片化的工具, detector 在脚下被默认值偷换, 再跑的数字与 b2 记录不可比。

## 2. 裁定: 有条件批准

**✅ 批准接线方向与最小 diff 设计**(config 两字段 + cli.py 默认读之, 引擎/检测器/dag/visualizer 零改动, config 一键回退)。**满足以下后 qw 隔离 worktree 实施, cc 验收:**

### 摸底更正 #0(必补)
把 `eval_tracking.py:34` 与 `diag_vehicle_track_fragmentation.py:174` 两处 no-param `cli.run` **显式钉死 v11+mask**(`crosswalk_detector=CrosswalkDetector(cfg), occ_denom="mask"`), 保住它们的 v11 baseline 语义(b2 记录口径), 与其它评测/诊断脚本一致遵加性。否则接线后这两个工具随生产默认漂移, b2 碎片化 baseline 报废。(若 qw/Jacob 反而希望这两个工具跟随生产 v2, 则需显式钉 v2+box 并在 [[b2-tracking-fragmentation-blindspot]] 注明 baseline 重立——cc 倾向前者: 先保口径。)

### 必改 #1: config 值词表对齐
方案用 `version: "v1"|"v2"`, 但 `eval_violations --detector` 的 choices 已是 `{"v11","v2"}`, 检测器类内部 version 也是 `cv-v11`。**同一检测器别造第三套名字**——config 用 `version: "v11"|"v2"`。并加一行注释: `version` 与既有 `crosswalk.method`(auto/segment/cv)正交——`method` 仅 v11(CrosswalkDetector)读, `version=="v2"` 时 CrosswalkDetectorV2 完全不读 method。

### 必改 #2: 护栏① 加确定性地板(determinism floor)
现方案「接线前跑一次存快照 → 接线后 v1 路径重跑 → 逐字节 diff」有歧义: 若 diff≠0, 分不清是接线 bug 还是管线固有 run-to-run 非确定性(torch CPU 线程等)。**必须先在当前 HEAD 连跑两次 v11+mask, 证明 run-to-run violations.csv 0 diff(确定性地板)**, 之后接线后的 diff 才能无歧义归因到接线。diff 范围钉 violations.csv(对, annotated.mp4 编码器噪声不比), 测试跑时 `annotated_video=False` 提速去噪。

### 必改 #3: 护栏③ 探针须含"持续单向平移(pan)"而非仅对称抖动
running-max 的**特定失效模式 = 持续漂移下的拖影**(镜头平移→斑马线移到新像素, 旧位置的 max 响应因永不衰减而残留→mask 被涂宽/错位)。±1~3% **对称**仿射抖动绕固定均值, 像素几乎不净移动, running-max 只会填出略膨胀的带(良性), **测不出拖影, 反给假安心**。探针**必须含一档单向平移/慢 pan, 幅度 ≥ 斑马线条纹宽在累积窗内的净位移**, 才真正压到该失效模式。对称抖动可留作对照。

## 3. cc 对 qw 两个开放问题的回答
- **合成探针够不够放行本次接线?** — **够, 放行**。cc 已核实 `input_video/` 现有 11 视频(9正+2负)全固定机位监控视角, 生产部署集 100% 无相机大幅运动 → running-max 拖影失效模式在生产中不触发。**真实运动样本 GT 是 Phase 2 阻塞项, 仅对"运动镜头部署"生效, 独立立项交 Jacob**。红线: 接线报告**不得宣称已泛化到运动镜头**——只可结论"固定机位部署安全 + 合成探针刻画了运动下的行为边界"。
- **非阻塞记录**: v2 的 `detect` 收 `vehicle_boxes` 但不使用(:44 预留), 即 v11 的车辆锚定加分路径在接线后休眠。**这不是隐藏回退**——C4 的 0.889 本就来自 sweep 用的无锚定 v2, 已反映此事实。

## 4. 实施顺序(cc 批准的流程)
1. qw 隔离 worktree(勿共享 HEAD)。
2. 改方案文档落 §2 的 #0/#1/#2/#3 四项。
3. 护栏①: 当前 HEAD 连跑两次 v11+mask → 证 run-to-run 0 diff(确定性地板)+ 存快照(含 sha256)。
4. 动代码: config 加 `version`/`occ_denom` 两字段 + cli.py 默认读之(#1 词表)+ #0 两脚本钉死 v11+mask。
5. 护栏①后测: config 回 `version:"v11"`+`occ_denom:"mask"` 重跑 → 与快照 11/11 逐字节 0 diff。
6. 护栏② 2×2 消融(v11+mask 复用快照 / v2+mask 新跑 / v11+box 新跑 / v2+box 复用 sweep), 同 harness, 拆 v2 时序 vs box 贡献。
7. 护栏③ 含单向 pan 的漂移探针 + 缺口结论(Phase 2 立项)。
8. 接线后默认(v2+box)端到端 11 视频 F1=0.889 复现。
9. 交付: 接线 commit + `docs/reports/2026-08-04-qw-crosswalk-wiring.md`。cc 逐项独立验收。

## 5. 红线
- cc 本次只读复核, 未改任何生产/qw 文件。
- 接线改动限 config.yaml + cli.py + #0 两脚本显式钉参; 引擎/检测器本体/dag/visualizer 零改动。
- scoped git add; qw 签 `Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>`。
- denom=box 语义变更(推翻 D2)已由 Jacob 拍板; 本关只审实现与护栏, 不复议方向。

---
**一句话**: qw 接线方案摸底逐点复核**准确**(接线点/接口/加性均对), 最小 diff 设计**批准**。但 cc 全仓枚举发现**影响面漏两个 no-param 调用方**(`eval_tracking.py` / `diag_vehicle_track_fragmentation.py` 会静默翻 v2+box, 污染 b2 碎片化 baseline)——必补显式钉死 v11+mask。另 3 项必改: config 值用 v11/v2 对齐既有词表; 护栏① 先立确定性地板再归因 diff; 护栏③ 探针须含单向 pan(对称抖动测不出 running-max 拖影)。合成探针**够放行本次固定机位部署**, 真实运动样本记 Phase 2, 报告不得宣称泛化到运动。四项落定后 qw 隔离 worktree 实施, cc 逐项验收。
