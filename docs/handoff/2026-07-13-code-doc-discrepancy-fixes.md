# Handoff — 代码/文档一致性修复 (第一批)

> 署名: **Claude Code** (协调者/裁决者)  ·  日期: 2026-07-13  ·  分支: `fix/code-doc-discrepancies`
> 交叉验证: Lingma 独立分析得出几乎相同的 P0 清单 (json import / very_loose / mode 默认 / viz denom / dag 未传车框)，已一并纳入本批修复。

## 目标
以**代码为准**核对工程,修掉一批"代码 ≠ 文档/注释/测试"的接线错误、崩溃点与默认值 footgun (不改识别算法骨架)。

## 已确认事实
- 环境: 本机 = **Apple M4 / arm64**。已建 `.venv`(homebrew **python3.11**, 已在 .gitignore),装了核心 CV 栈(**cv2 5.0.0 / numpy 2.4.6 / pyyaml / tqdm / pytest**)。torch/torchvision/ultralytics 后台安装中(网络慢);hyperlpr3 暂未装(可选, 代码有 `_HAS_HL` 降级)。系统默认 `python3` 是 3.7.5(太老), **务必用 `.venv`**。
- 已让 `geometry.py` 的 `cv2` 改为**函数内延迟导入** → 纯逻辑测试在无 OpenCV 机也能跑。
- **本机全量测试已全绿**(装了 cv2 后): `source .venv/bin/activate && python -m pytest tests/ -q` → **64 passed**;`python scripts/run_tl_tests.py` → **10/10**。含 cv2 的 geometry/traffic_light/dag 集成测试也已在 Mac 验证通过。
- 集成/单测中 **3 条陈旧夹具**(D2 分母=mask 迁移遗留, box 尺寸按旧 box-分母写)已按 mask 分母+footprint=0.5 重算: `test_violation_engine_v2` 2 条 + `test_dag` 1 条。它们在 HEAD 上本就失败(先被 mode footgun 挡在前面,没暴露)。
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
**本机测试已全绿, 剩真实模型冒烟。** 待后台 torch/ultralytics 装完, 在 `.venv` 里跑一次真实推理冒烟:
`source .venv/bin/activate && python scripts/run_video.py input_video/违章02.mp4 --preset balanced`
——确认默认即 pedestrian_green 语义、`--cot` 不再崩、YOLO 权重按 `models/yolov8n.pt` 加载。
之后再由用户 push 分支(此环境网络对 github 超时, push 需在有网机器执行), 供 Windows 端同步; 合 `main` 前建议 Windows 也跑一遍(hyperlpr3 真实车牌栈)。

> 状态更新(2026-07-13, Claude Code): P0/P1 修复 + 3 条陈旧夹具已在 Mac(.venv, cv2)全量验证通过。

## 多 agent 协作事件 + 署名更正 (2026-07-13, Claude Code 记录)
- **MOTA 指标(M6, design v2 §5.1 缺口)由 Lingma 实现**: `evaluation/metrics.py` 的 `mota_metrics` + `evaluator.py` + `tests/unit/test_metrics.py` 的 MOTA 用例。**功劳归 Lingma。**
- **协作事故**: Lingma 在**共享工作树**里边改 MOTA(未提交), 我(Claude Code)用 `git add -A` 提交自己的改动时**误把 Lingma 的 MOTA 未提交改动扫进了 commit `78dfe38` / `d7a5c72`**, 导致其工作被并入我的 commit、未正确署名。Lingma 的 handoff(`2026-07-13-lingma-full-analysis.md`)也在两次提交间被创建又删除(其主动让位给本权威 handoff)。
- **处置**: 分支未 push、全绿(74 passed), **不重写历史**(Lingma 可能仍在改, 重写 tip 会打乱其工作树); 以此节更正署名。
- **规则(所有 agent 遵守)**: 本仓库**禁止 `git add -A`/`git add .`**, 只 `git add <明确路径>` 且提交前 `git status` 确认无他人在途改动被误纳。
- 当前测试基线: `.venv` 下 `pytest tests/` = **74 passed**(含 Lingma 的 MOTA)。

## 会话进展 + 第二次(更严重)协作事故 (2026-07-13 晚, Claude Code)
- **M1 Phase1 运行时管线已完成**(计划 `docs/plans/2026-07-13-m1-phase1-runtime-plumbing.md`): T1 候选并集(IoMin去重)/ T2 状态分类器 ONNX 包装(优雅回退)/ T3 vehicle 暴露 COCO traffic-light 框 / T4 traffic_light `ped_classifier` 编排分支 / T5 dag 路由灯框 / T6 config。**全套 pytest = 85 passed, run_tl_tests 10/10**。无权重时自动回退 color 路径, 可安全合入且不改现有行为。Phase2(数据+训练产出 `models/ped_signal.onnx`)/ Phase3(LOVO + prior-free 评测)待做。
- **⚠️ 严重事故: 多 agent 共享同一工作树 + 同一 HEAD。** Lingma(eval 重构)在共享工作树 `git checkout` 到 `main` 再到新分支 `refactor/eval-common-modules`, **把我的 checkout 一起切走**, 丢失了未提交的 T4 编辑(已在隔离 worktree 重做); 已提交工作全部安全。
- **处置 + 规则**: 我已迁到独立 worktree `/Users/jacob/personal/crosswalk-guard-fix`(分支 `fix/code-doc-discrepancies`)继续。**所有 agent 今后各用独立 git worktree 或 clone**; 主目录随时可能在别的 agent 分支上。
- **分支现状**(均基于 `ae115d3`, 待有网机器 push): `fix/code-doc-discrepancies`(我: 一致性修复 + Lingma MOTA + M1 spec/plan + M1 Phase1); `refactor/eval-common-modules`(Lingma: eval 共用模块)。建议合并顺序: 先合 fix 分支, Lingma 重构再 rebase 其上。

## 分支合并 + M1 Phase2 工具链 (2026-07-13 深夜, Claude Code)
- **已全部合并到 `main` 并转 trunk-based 开发**(用户定策): `fix` FF 进 main; `refactor/eval-common-modules` 3-way 零冲突合入; 联网后 `feat/light-eval-v2`(另一 agent/WorkBuddy: color 路径 prior 重标定 + 迟滞 + 灯态GT/回归)也合入(**剔除其强制入库的 gitignore 生成物 `data/output/light_eval/*`**)。临时分支已删(本地+远程), 只剩 `main`。合并暴露的 Lingma `image_utils` 2 红测已修(空图→False/缺失→None)。
- **M1 Phase2 工具链完成(数据到位即可训, 本机已合成数据验证契约)**:
  - `src/redlight/data_pipeline/ped_signal_dataset.py`: 纯核心 `light_state_to_label`/`assign_crop_labels`(先验挑信号其余off)/`crop_box`/`lovo_folds`/`load_labeled_crops` + `extract_crops`。8 单测。
  - `scripts/build_ped_signal_crops.py`: 抽候选ROI+弱标签→`datasets/ped_signal/`(**版本化训练数据, 非 gitignore**)。真实 GT+先验冒烟通过。
  - `scripts/train_ped_signal.py`: tiny CNN→LOVO→`torch.onnx`(dynamo=False)导出 `models/ped_signal.onnx`。**`--smoke` 合成数据契约自检通过**(train acc 1.00 → ONNX → cv2.dnn/SignalStateClassifier 分类 [walk,stand,off] 正确)。
  - pyproject 加 `train` extra(torch+onnx); 运行时只需 cv2.dnn。本机 `.venv` 另装 onnx 1.22。
- **当前测试基线: `pytest tests/` = 145 passed, run_tl_tests 10/10**(本机 .venv)。
- **⚠️ 数据卡点**: 本机无源视频/预抽帧(`datasets/frames/` gitignore, 只在 Windows; 本机仅 30 张未标 light_eval crop)。`datasets/ped_signal/` 尚空。

### 下一步动作 (唯一首要)
在有视频/帧的机器上产出真实 crop 数据集, 三步:
1. `python scripts/build_ped_signal_crops.py --frames-dir datasets/frames`(或 `--video-dir input_video`)→ 生成 `datasets/ped_signal/{crops, labels.csv}`。
2. 灯态画廊人工校验 crop 标签置 `verified=1`(尤其无先验的 01/05/06/07/08/09/11)。
3. `python scripts/train_ped_signal.py` → 看 LOVO 平均 acc + 导出 `models/ped_signal.onnx`；把 `configs/config.yaml` 的 `traffic_light.method` 设 `ped_classifier` 即启用。
`datasets/ped_signal/` 建议 commit 进 git(小, 训练数据), 跨机 pull 复用, 免再传大视频。
> 备注: main 有未 push 提交(网络间歇性超时); 联网稳定后 `git push origin main`。
