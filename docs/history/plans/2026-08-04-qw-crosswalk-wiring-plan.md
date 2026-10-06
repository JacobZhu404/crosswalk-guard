# 斑马线 v2 接线方案(qw, C3 — Jacob 拍板后的实施计划)

> Jacob 拍板:**接线** = v2 时序聚合 + denom=box + box_overlap=0.20 接 `cli.run` 默认, 推翻 D2(mask 分母)语义。
> cc 已定三护栏验收(773dff4 §4): ① 不接线默认路径 bit-identical 回归 ② 消融拆 v2 时序 vs denom=box 贡献 ③ 相机大幅运动样本测 running-max 漂移泛化。
> **cc 已批准本方案(`docs/handoff/2026-08-04-cc-gate-qw-crosswalk-wiring-plan.md`), 附 4 项落定修正(见 §0)。qw 在隔离 worktree 实施, cc 逐项独立验收。**

## 0. cc gate 4 项落定修正(2026-08-04, 全部并入实施)

1. **影响面修正(qw 摸底漏 2 个 no-param 调用方)**: 除 run_video.py/cli main 外, 以下两脚本也调 `cli.run` 不传参, 接线后会被静默翻到 v2+box, **破坏度量同口径红线**:
   - `scripts/diag_vehicle_track_fragmentation.py:174` — 用 confirmed 事件算碎片化, v2 改变 confirm 集合 → b2 碎片化 baseline(在 v11+mask 下测的)直接漂移
   - `scripts/eval_tracking.py:34` — 全管线在 v2+box 下重跑
   **必补: 这两处显式钉死 v11+mask(传 `CrosswalkDetector(cfg)` + `occ_denom="mask"`)。**
2. **config 词表**: `crosswalk.version: "v11"/"v2"`, 对齐 `eval_violations.py` 既有 `--detector` 词表(不造第三套名); 注释声明 version 与既有 `method` 字段正交(互不影响)。
3. **护栏①先立确定性地板**: 在接线前(当前 HEAD)把默认路径(v11+mask)连跑两次, 证 run-to-run 0 diff; 再跑接线后 `version="v11"` 回退路径与快照 diff —— 否则 diff≠0 分不清是接线 bug 还是 torch 固有非确定性。
4. **护栏③探针必须含单向 pan 档**: running-max 的失效模式是"持续漂移拖影"(mask 面积单调膨胀), ±1-3% 对称抖动像素几乎不净移动测不出 → 探针三档: 静止对照 / 对称抖动 / **单向 pan(持续漂移)**。

cc 已代回两个开放问题(不阻塞):
- 合成探针够放行本次接线(现有 11 视频全固定机位监控, 拖影生产不触发); **真实运动样本记 Phase 2 独立立项交 Jacob**; 接线报告不得宣称已泛化到运动镜头。
- v2 不用 vehicle_boxes → v11 车辆锚定路径休眠, 非隐藏回退(0.889 本就来自无锚定 v2, sweep 用的就是它)。

## 1. 接线点(最小 diff, 全部在 config.yaml + cli.py)

### 现状(已核实)
- `cli.py:75`: `"crosswalk": crosswalk_detector if crosswalk_detector is not None else CrosswalkDetector(cfg)` → 默认 v11
- `cli.py:71-72`: `occ_denom` 仅显式传时进 `_engine_kwargs`; 否则 None → 引擎默认 `"mask"`(D2 现状)
- `cli.py:80`: `Visualizer(cfg, preset=preset, occ_denom=occ_denom or "mask")`
- `dag.py:69` `cw_int=crosswalk_interval=4` → 每 4 采样帧调一次 `cw.detect(frame, vb)`(生产节奏)
- v11/v2 的 `detect(frame, vehicle_boxes=None)` 签名一致, 均返回二值掩膜 (h,w) uint8 → **接口天然兼容**
- v2 `_accum` running-max 状态每视频新建(cli.run 每视频新建 comp) → 无跨视频污染
- `tracker.py` SENSITIVITY_PRESETS 四档已含 `box_overlap`; `violation_engine.py` 已按 `occ_denom=="box"` 选 `box_overlap`; `visualizer.py` denom 与阈值与引擎同源 → **引擎/可视化层零改动**

### 改法(纯加性, 参数优先)
1. `configs/config.yaml` `crosswalk:` 段新增两字段:
   - `version: "v11"`  # 斑马线检测器词表: v11(现状) | v2(时序聚合), 与 eval_violations --detector 对齐; 与 method 正交
   - `occ_denom: "box"` # 占道分母: mask(D2 现状) | box(接线)
2. `cli.py`:
   - 默认 detector: `crosswalk_detector if crosswalk_detector is not None else (CrosswalkDetectorV2(cfg) if getattr(cfg.crosswalk, "version", "v11") == "v2" else CrosswalkDetector(cfg))`
   - 默认 occ_denom: `occ_denom if occ_denom is not None else getattr(cfg.crosswalk, "occ_denom", "mask")`(进 `_engine_kwargs` 与 Visualizer)
3. 不动 `tracker.py` / `violation_engine.py` / `dag.py` / `visualizer.py`。

**影响面(已核实)**: 显式传参调用方(eval_violations.py `--detector v11 --occ-denom mask` 默认、sweep、各 diag)参数优先,**输出不受影响**; 受影响仅"不传参"的生产入口(`run_video.py` / `cli main`)。

**回退开关**: config 两字段改回 `version: "v11"` + `occ_denom: "mask"` 即恢复接线前行为(护栏①验证的正是此路径 bit-identical)。

### 语义变更声明(推翻 D2)
occ_denom mask→box: 占道比例分母从"车落进斑马线 mask 的面积/整车 footprint"改为"车 footprint 落在斑马线内的比例"; 阈值同取 balanced 0.20(数值与 overlap 相同, 但分母语义不同, 由 0.609→0.889 端到端实证支撑, 773dff4)。

## 2. 三护栏执行计划

### ① bit-identical 回归(不接线默认路径)
- **第一步(接线前, 当前 HEAD): 连跑两次** 不传参 `cli.run`(= v11+mask 现状默认) 全 11 视频 → `data/output/qw/wiring_pre/run1|run2/` 存 violations.csv + sha256 清单 → **run1 vs run2 必须 0 diff = 确定性地板**(先排除 torch 固有非确定性)
- 第二步(接线后): config `version="v11"` + `occ_denom="mask"` 再跑同 11 视频 → 与 run1/run2 快照 **bit-level diff**(逐字节)
- 通过标准: 地板 0 diff ∧ 回退路径 11/11 视频 0 diff → 接线 diff 完全归因于接线本身
- 说明: v11 路径代码零改动, 仅 config 分支读回 v11, 预期逐字节一致; 若地板本身非 0, 先归因非确定性(不掩盖, 如实记录)

### ② 2×2 消融(拆 v2 时序聚合 vs denom=box 对 +46% 的贡献)
| 组合 | 数据来源 |
|---|---|
| v11+mask(基线) | 护栏① run1 快照(复用, 已 0-diff 双跑) |
| v2+mask | 新跑 11 视频 |
| v11+box | 新跑 11 视频 |
| v2+box(0.20) | sweep 已有 55 份数据(5 档全 F1=0.889, 复用 box_overlap=0.20) |

- harness: `eval_violations.py` 同口径(`cli.run` + `match_violation_events(min_overlap_s=0.5)`), 禁 --reuse
- 交付: 4 组事件级 P/R/F1 表 + 贡献拆解(v2 时序清碎片 FP/救违章11 vs box 提 P)
- 预期: v2 维度=清 6 碎片 FP + 救回违章11(773dff4 §0(B)); box 维度=提 precision(mask 过宽稀释, denom=box 缓解)

### ③ 相机大幅运动样本(running-max 漂移泛化)
- **样本现状(已核实)**: `input_video/` 现有 11 视频(9正+2负)全为固定机位监控视角, **无真实相机大幅运动样本** → 真实样本记 **Phase 2 独立立项交 Jacob**, 本接线**不宣称已泛化到运动镜头**
- **本次交付探针(合成漂移, 三档)**:
  - 静止对照: 原视频直接跑 v2(基线行为)
  - 对称抖动: 每帧仿射 ±1~3% 缩放/平移/旋转(像素净位移≈0)
  - **单向 pan(cc 修正必含)**: 持续单向平移/缩放(累计净漂移 > 帧宽 5%~15%), 模拟镜头漂移 —— running-max 失效模式是**持续漂移拖影**(mask 面积单调膨胀)
  - 测: (a) v2 mask 面积/质心时间序列(拖影=面积持续增长) (b) 三档 mask-IoU vs 静止对照(用现成 GT poly 对 warp-back mask) (c) 事件级是否产生假斑马线/假违规
- 代表视频: 取 2-3 个(如 02/06/08, mask-IoU 高中低各一)

## 3. 验收标准(cc 可验)
- 护栏①: 接线前 run1 vs run2 确定性地板 0 diff ∧ 接线后 `version="v11"` 回退 11/11 视频 0 diff
- 护栏②: 4 组 P/R/F1 表 + 贡献拆解, 与 773dff4 基线逐视频吻合
- 护栏③: 三档(静止/抖动/单向 pan)合成探针结果 + 真实运动样本缺口结论(Phase 2 立项, 不宣称已泛化)
- 最终: 接线后默认路径(v2+box)端到端 11 视频 F1=0.889 复现(与 cc 重建一致)

## 4. 风险与回退
- **回退**: config 一键回 v11+mask; 显式传参评测路径从未受影响
- **性能**: v2 每 detect 全帧 Sobel×1 + running-max 聚合, 与 v11(多条带扫描)同量级; 实测对比附在护栏②报告
- **违章01 FP / 违章04 FN**: 既有老问题(C4 已证非 v2 引入), 不归本接线
- **v2 mask 首帧弱**: 生产需持续 5 帧(0.625s), 前几帧弱积累不影响判定(before/after 报告已注)

## 5. 红线
- 不碰 datasets/gt、不碰 wb 输出目录/缓存; scoped git add; Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>
- 接线改动仅 config.yaml + cli.py; 引擎/检测器本体零改动
- cc 审后实施; 实施后交付物: 接线 commit + 三护栏报告(docs/reports/2026-08-04-qw-crosswalk-wiring.md)
