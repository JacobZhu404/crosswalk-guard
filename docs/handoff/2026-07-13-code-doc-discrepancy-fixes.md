# Handoff — 代码/文档一致性修复 (第一批)

> 署名: **Claude Code** (协调者/裁决者)  ·  日期: 2026-07-13  ·  分支: `fix/code-doc-discrepancies`
> 交叉验证: Lingma 独立分析得出几乎相同的 P0 清单 (json import / very_loose / mode 默认 / viz denom / dag 未传车框)，已一并纳入本批修复。

## 目标
以**代码为准**核对工程,修掉一批"代码 ≠ 文档/注释/测试"的接线错误、崩溃点与默认值 footgun (不改识别算法骨架)。

## 已确认事实
- 环境: 本机 macOS / Python 3.7.5, **无 cv2/torch/ultralytics/hyperlpr3** → 主流程与依赖 cv2 的测试**无法在本机跑**,须 Windows 端验证。
- 已让 `geometry.py` 的 `cv2` 改为**函数内延迟导入** → 纯逻辑测试(engine/tracker/metrics)现可在无 OpenCV 的编码机运行。
- **45 个纯逻辑单测全绿** (`PYTHONPATH=src python3 -m pytest tests/unit/test_violation_engine_v2.py tests/unit/test_tracker_v2.py tests/unit/test_metrics.py tests/unit/test_light_metrics.py -q`)。
- 关键裁决: `ViolationEngineV2` 之前**默认 `mode="red_light"`(已废弃的旧语义)**,直接导致引擎单测(test_green_confirmed/test_red_not_violation 等)在 HEAD 上**本就失败**。这是 plate-agent 恢复双模式与 traffic-light-agent 测试/设计 v2 的冲突。**按权威规格(design v2 + E12/E13)裁决: 删除双模式, 锁定 pedestrian_green。**
- `test_loose_catches_partial_overlap` / `test_unknown_occluded_to_review` 两条 fixture 是 **D2(分母=mask)迁移遗留的陈旧夹具**(注释仍写 box 分母 0.25),与我的改动无关;已按 mask 分母+footprint=0.5 重算夹具,保持原测试意图。

## 已排除项 (勿重复踩)
- **勿把默认语义留成 red_light / 勿恢复 mode 双模式开关** — 这是 E13 footgun,已彻底删除,不要再加回。
- **勿改红绿灯识别算法**(候选/聚类/锚点/lamp_score) — 本批不碰;对比度门控、frac_v>=0.30、v6 单轨迹选灯、全场景像素求和均已被前序实证否决(见 handoff 07 第 7 节)。
- **省份先验"京"实际放行 京/冀/津/晋** 是 OCR 混淆别名(`_PROVINCE_ALIASES`),**故意为之**,本批不改;仅需知道 `handoff_plate_optimization.md` 里"仅京"的说法不准确。

## 修改文件清单
| 文件 | 变更 |
|---|---|
| `src/redlight/pipeline/violation_engine.py` | 删除 `mode` 参数与 red_light 分支,锁定 pedestrian_green;更新模块 docstring |
| `src/redlight/app/cli.py` | 补 `import json`(修 `--cot` 崩溃);`run()` 去掉 `mode`;argparse `--preset` 补 `very_loose`;`Visualizer(cfg, preset=preset)` |
| `scripts/run_video.py` | 去掉 `--mode` 参数 |
| `scripts/debug_{57s,dag_mask,overlap,violation,target_frame}.py` | `ViolationEngineV2("loose")`(去 mode) |
| `src/redlight/pipeline/dag.py` | `n_crosswalk` 传入 `vehicle_boxes`(激活 v11 车辆锚定) |
| `src/redlight/pipeline/visualizer.py` | 与引擎对齐: `footprint=0.5, denom="mask"` + preset.overlap 阈值 |
| `src/redlight/pipeline/analysis.py` | COT 修键: `cls`(非 vehicle_class) / `avg_conf`(非 conf) |
| `src/redlight/models/vehicle.py` | 读 `cfg.models.vehicle`(相对路径按工程根解析),不再找不存在的 `vehicle_pt` |
| `src/redlight/infrastructure/geometry.py` | `cv2` 延迟导入(纯逻辑可测) |
| `tests/unit/test_violation_engine_v2.py` | 两条 D2 陈旧夹具按 mask 分母重算 |
| `configs/config.yaml` / `pyproject.toml` | 注释 v6→v7-stable;版本 2.0.0→2.3.0 |

## 约束条件
- 跨双机(Mac 编码 / Windows 跑推理),依赖须通用;committed 代码勿硬编码绝对路径(用 `project_root()`)。
- pytest 因 cv2 慢会 hang → 红绿灯单测用 `scripts/run_tl_tests.py`;本机跑纯逻辑测试须带 `PYTHONPATH=src`(旧 pytest 不认 pyproject 的 `pythonpath`)。
- 多 agent 协作: 署名、勿互相覆盖;冲突由 Claude Code 依权威规格裁决。

## 下一步动作 (唯一首要)
**在 Windows(有 cv2/torch/ultralytics/hyperlpr3)拉取本分支, 跑三处验证:**
1. `python scripts/run_tl_tests.py`(红绿灯 10/10 应仍过);
2. `python -m pytest tests/ -q`(含 cv2 的 geometry/traffic_light/dag 集成测试);
3. `python scripts/run_video.py input_video/违章02.mp4 --preset balanced` 冒烟,确认默认即 pedestrian_green 语义 + `--cot` 不再崩。
通过后再合入 `main` 并 push,供其他机器同步。
</content>
