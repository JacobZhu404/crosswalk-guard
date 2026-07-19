# Phase 2 旧GT-F1 vs 新GT-F1 对比

> 生成于 Phase 2 重建（build_gt.py --write）。本文件不被脚本覆盖，作为测量尺变更透明的留档。

## 方法

Phase 2 决策 2 只提交 `light_states.csv` 的 merge（crosswalk 仅 dry-run，不提交）。
为验证「测量尺未移动」，对重建前后的 light canonical 做**段内容字段级比对**：

- 旧 GT：`_snapshot_pre_build/light_states.csv`（pre-build 快照，亦可由 `git show gt-pre-phase2:datasets/gt/light_states.csv` 取得，二者一致）
- 新 GT：当前 `datasets/gt/light_states.csv`（build_gt.py --write 覆盖后）
- 比对：逐视频逐段比较 `(start_s, end_s, state, confidence, note)`，**note 先剥去 CSV 引号**（兼容历史畸形未引号逗号行），避免把引号风格差异误判为内容差。

## 结果

| 维度 | 旧 GT (pre-build) | 新 GT (post-build) |
|---|---|---|
| 视频数 | 11 | 11 |
| 段总数 | 24 | 24 |
| 段内容字段级差异 | — | **0 差异** |

唯一字节级差异：pre-build 中 1 行（违章01 `[48,62.5]` 的 occluded 段）note 含**未加引号的逗号**（历史畸形 CSV），重建后该 1 行被规范为合法 CSV（note 字段正确加引号）。**note 文本零丢失**，段内容完全一致。

## 结论

**旧GT-F1 == 新GT-F1**。light canonical 段内容逐字段 0 差异，故任何以该 canonical 为入参的指标（灯态 F1、违章事件 F1、追踪 F1）数学上不变；既有基线（灯态评测 0.889 等）全部保留。

补充：
- `eval_violations.py` 的 GT 来自 `events.csv`（不读 `light_states.csv`），故违章事件 F1 与本次 light merge 无关，无论如何不受影响。
- 全流水线重跑（11 视频）仅会重新确认上述基线、且耗时长、需视频+模型资源；如需实跑数字，可在有缓存（`--reuse`）时按需执行，此处以「段内容 0 差异 ⇒ 指标不变」完成透明闭环。
