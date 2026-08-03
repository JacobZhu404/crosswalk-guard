# 斑马线 v2 时序聚合 before/after 报告(qw, B1 实现验证)

> B1 方案(`docs/plans/2026-08-03-qw-crosswalk-fix-plan.md`)实现后的 mask-IoU before/after 对比。
> 核心改动: `eval_crosswalk_mask.py` 加 `--temporal` 模式, v2 走完整视频 running-max 时序聚合(镜像生产节奏 8fps×interval4)。
> cc 裁定条件 C1(镜像生产节奏)、C2(纯加性)、C4(mask-IoU 升) 验证。

## before/after 对比

| 指标 | v11(基线) | v2 单帧(before) | v2 时序聚合(after) | 提升 |
|---|---|---|---|---|
| mask-IoU 均值 | 0.011 | 0.249 | **0.441** | +0.192 (+77%) |
| recall 均值 | 0.012 | 0.385 | **0.818** | +0.433 (+113%) |
| precision 均值 | 0.093 | 0.452 | **0.506** | +0.054 (+12%) |
| IoU≥0.5 的视频数 | 0/9 | 0/9 | **3/9** | +3 |

**结论**: 时序聚合将 mask-IoU 从 0.249 抬到 0.441, 与 v6 改定 1 预测的 train-free 天花板 ~0.4 精确吻合。recall 从 0.385 翻到 0.818, 说明 running-max 成功恢复了被遮挡/漏检的条纹。precision 略升但仍是瓶颈(0.506), 因为梯形 mask 面积偏大溢出 → denom=mask 稀释效应仍在, 但 denom=box 可缓解(历史 0.941 vs 0.875)。

## 逐视频 after(v2 时序聚合)

| 视频 | after IoU | after recall | after precision | ≥0.5? |
|---|---|---|---|---|
| 违章02 | 0.406 | 0.721 | 0.447 | ❌ |
| 违章03 | 0.430 | 0.856 | 0.475 | ❌ |
| 违章04 | 0.248 | 0.773 | 0.282 | ❌ |
| 违章05 | 0.364 | **1.000** | 0.364 | ❌ |
| 违章06 | **0.646** | 0.965 | 0.664 | ✅ |
| 违章07 | 0.438 | 0.531 | 0.717 | ❌ |
| 违章08 | **0.527** | 0.918 | 0.561 | ✅ |
| 违章09 | 0.364 | 0.834 | 0.404 | ❌ |
| 违章11 | **0.550** | 0.787 | 0.658 | ✅ |

关键观察:
- 违章05 recall=1.000(全量覆盖)但 precision=0.364(mask 面积太大) → denom=box 更适合此视频
- 违章06 表现最好(IoU=0.646, recall=0.965, precision=0.664)
- 违章04 最差(IoU=0.248), 但 recall=0.773 说明梯形位置对, 只是 mask 过宽(precision=0.282)
- 首帧(ts=1.0s)普遍偏低因为 accumulator 刚启动(1-2 帧); 生产中违章需持续 5 帧(0.625s), 前几帧弱积累不影响判定

## cc 验收对照(C1–C4)

- **C1 镜像生产节奏**: ✅ `--temporal` 模式按 `cfg.inference.fps`(8fps)×`interval`(round(fps/8)=4) 采样, 与 DAG `n_crosswalk` 每 4 帧调一次 `detect` 完全一致
- **C2 纯加性**: ✅ v11 路径(无 `--temporal`)每帧独立实例, 口径不变; v2 单帧模式也保持原口径; `box_overlap` 已在 `tracker.py`/`violation_engine.py` 落地, 默认 denom=mask 路径用 `overlap` 阈值(不变), 仅 denom=box 时用 `box_overlap`
- **C3 不接线**: ✅ v2 走注入(`cli.run(crosswalk_detector=CrosswalkDetectorV2(cfg), occ_denom="box")`), 默认仍 v11; Jacob 拍板才接线
- **C4 mask-IoU 升**: ✅ 0.249→0.441(+77%), recall 0.385→0.818(+113%), 3/9 视频≥0.5; 触顶 ~0.4 与 v6 改定 1 预测吻合

## 触顶判断

v6 改定 1 预测 train-free 触顶 ~0.4(车底 60% 遮挡外推不了)。实测 0.441 略超预期, 但 6/9 视频 <0.5, 瓶颈是 precision(mask 过宽)而非 recall(0.818 已很高)。可能的后续优化:
- 缩小梯形 mask 宽度(收紧 5-95 百分位 → 10-90) → 提 precision
- denom=box(已就绪) → 绕过 mask_area 稀释
- 若仍触顶, 记 Phase 2 seg 微调为独立立项交 Jacob

## 交付物

- `scripts/eval_crosswalk_mask.py` — 加 `--temporal` 模式 + recall/precision 输出
- `scripts/sweep_box_overlap.py` — box_overlap 阈值扫描(待跑)
- `tests/unit/test_no_gt_leakage.py` — 2/2 pass(GT 不进生产)
- 本报告

## 下一步(待 cc 放行)

- 跑 `sweep_box_overlap.py` 定最优 `box_overlap` 阈值(9正+2负, 5档扫描)
- 跑 `eval_violations.py` 端到端 F1(v2 + denom=box + 新阈值)
- 检查负例 01/10 零新误报
