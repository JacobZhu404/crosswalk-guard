# 斑马线 v2 接线方案(qw, C3 — Jacob 拍板后的实施计划, 待 cc 审)

> Jacob 拍板:**接线** = v2 时序聚合 + denom=box + box_overlap=0.20 接 `cli.run` 默认, 推翻 D2(mask 分母)语义。
> cc 已定三护栏验收(773dff4 §4): ① 不接线默认路径 bit-identical 回归 ② 消融拆 v2 时序 vs denom=box 贡献 ③ 相机大幅运动样本测 running-max 漂移泛化。
> 本方案仅列接线点与护栏执行计划, **不实施**; cc 审后开干。

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
   - `version: "v2"`   # 斑马线检测器: v1(v11 现状) | v2(时序聚合)
   - `occ_denom: "box"` # 占道分母: mask(D2 现状) | box(接线)
2. `cli.py`:
   - 默认 detector: `crosswalk_detector if crosswalk_detector is not None else (CrosswalkDetectorV2(cfg) if getattr(cfg.crosswalk, "version", "v1") == "v2" else CrosswalkDetector(cfg))`
   - 默认 occ_denom: `occ_denom if occ_denom is not None else getattr(cfg.crosswalk, "occ_denom", "mask")`(进 `_engine_kwargs` 与 Visualizer)
3. 不动 `tracker.py` / `violation_engine.py` / `dag.py` / `visualizer.py`。

**影响面(已核实)**: 显式传参调用方(eval_violations.py `--detector v11 --occ-denom mask` 默认、sweep、各 diag)参数优先,**输出不受影响**; 受影响仅"不传参"的生产入口(`run_video.py` / `cli main`)。

**回退开关**: config 两字段改回 `version: "v1"` + `occ_denom: "mask"` 即恢复接线前行为(护栏①验证的正是此路径 bit-identical)。

### 语义变更声明(推翻 D2)
occ_denom mask→box: 占道比例分母从"车落进斑马线 mask 的面积/整车 footprint"改为"车 footprint 落在斑马线内的比例"; 阈值同取 balanced 0.20(数值与 overlap 相同, 但分母语义不同, 由 0.609→0.889 端到端实证支撑, 773dff4)。

## 2. 三护栏执行计划

### ① bit-identical 回归(不接线默认路径)
- 接线前(改代码前, 即当前 HEAD): 跑不传参 `cli.run`(= v11+mask 现状默认) 全 11 视频 → `data/output/qw/wiring_pre/` 存 violations.csv + sha256 清单
- 接线后: config `version="v1"` + `occ_denom="mask"` 再跑同 11 视频 → 与快照 **bit-level diff**(逐字节)
- 通过标准: 11/11 视频 violations.csv 0 diff
- 说明: v1 路径代码零改动, 仅 config 分支读回 v11, 预期逐字节一致; 若浮点/顺序差异暴露, 如实记录并定位(不掩盖)

### ② 2×2 消融(拆 v2 时序聚合 vs denom=box 对 +46% 的贡献)
| 组合 | 数据来源 |
|---|---|
| v1+mask(基线) | 护栏①接线前快照(复用) |
| v2+mask | 新跑 11 视频 |
| v1+box | 新跑 11 视频 |
| v2+box(0.20) | sweep 已有 55 份数据(5 档全 F1=0.889, 复用 box_overlap=0.20) |

- harness: `eval_violations.py` 同口径(`cli.run` + `match_violation_events(min_overlap_s=0.5)`), 禁 --reuse
- 交付: 4 组事件级 P/R/F1 表 + 贡献拆解(v2 时序清碎片 FP/救违章11 vs box 提 P)
- 预期: v2 维度=清 6 碎片 FP + 救回违章11(773dff4 §0(B)); box 维度=提 precision(mask 过宽稀释, denom=box 缓解)

### ③ 相机大幅运动样本(running-max 漂移泛化)
- **样本现状扫描(已核实)**: `input_video/` 现有 11 视频(9正+2负)全为固定机位监控视角, **无相机大幅运动样本** → 真实运动样本 GT 是**缺口**
- 本次交付内可做: **合成漂移探针** —— 对现有固定机位视频加仿射抖动(±1%~3% 缩放/平移/旋转, 多档), 跑 v2 测: (a) mask-IoU vs 原静止机位(用现成 GT poly) (b) running-max 累积下 mask 面积/位置漂移量 (c) 事件级是否产生假斑马线/假违规
- 真实运动样本: 记 **Phase 2 独立立项交 Jacob**(采集运动镜头 + GT 标注 + 正式评估), 不阻塞本次接线(cc 可裁定: 合成探针结论是否够本次放行, 或要求样本到位后再终判)

## 3. 验收标准(cc 可验)
- 护栏①: 11/11 视频 bit-identical(0 diff)
- 护栏②: 4 组 P/R/F1 表 + 贡献拆解, 与 773dff4 基线逐视频吻合
- 护栏③: 合成漂移探针结果 + 缺口结论(Phase 2 立项)
- 最终: 接线后默认路径(v2+box)端到端 11 视频 F1=0.889 复现(与 cc 重建一致)

## 4. 风险与回退
- **回退**: config 一键回 v1+mask; 显式传参评测路径从未受影响
- **性能**: v2 每 detect 全帧 Sobel×1 + running-max 聚合, 与 v11(多条带扫描)同量级; 实测对比附在护栏②报告
- **违章01 FP / 违章04 FN**: 既有老问题(C4 已证非 v2 引入), 不归本接线
- **v2 mask 首帧弱**: 生产需持续 5 帧(0.625s), 前几帧弱积累不影响判定(before/after 报告已注)

## 5. 红线
- 不碰 datasets/gt、不碰 wb 输出目录/缓存; scoped git add; Co-Authored-By: 千问办公 <qw@crosswalk-guard.agents>
- 接线改动仅 config.yaml + cli.py; 引擎/检测器本体零改动
- cc 审后实施; 实施后交付物: 接线 commit + 三护栏报告(docs/reports/2026-08-04-qw-crosswalk-wiring.md)
