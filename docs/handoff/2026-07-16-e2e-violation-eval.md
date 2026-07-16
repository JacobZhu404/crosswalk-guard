# Handoff — 端到端违章事件评测(eval-e2e)落地 + 首个基线 (2026-07-16)

> 署名: Claude Code · 日期: 2026-07-16

## 目标
补齐 README/CHANGELOG 承诺却缺失的"事件级 P-R-F1"(Task#10): 建端到端违章评测, 把整条流水线的
最终违章结论与 `datasets/gt/events.csv` 对比, 作为后续修复的客观标尺。

## 已交付
- `src/redlight/evaluation/violation_eval.py`(纯函数, 可单测): 违章段GT加载 + 事件级重叠贪心匹配
  (P/R/F1) + 覆盖率诊断(暴露欠检) + 车牌次级指标 + 多视频聚合。
- `tests/unit/test_violation_eval.py`: 11 项单测(重叠/漏检/误报/覆盖/车牌/聚合), 全绿。全 unit 216 项全绿。
- `scripts/eval_violations.py`: 跑流水线(或 --reuse 读已有 violations.csv) -> 逐视频时间线 + 总体指标。
  用法 `python scripts/eval_violations.py [--videos ...] [--reuse] [--min-overlap 0.5]`。

## 主指标口径(用户选定)
**事件级重叠匹配**: 预测 confirmed 事件与 GT is_violation=1 段按时间重叠(≥min_overlap_s, 默认0.5s)
贪心 1:1 配对 -> TP/FP/FN。辅助 **覆盖率** = GT违章段被预测事件覆盖的时长占比(暴露"抓到但只抓一小段")。
车牌为**次级**指标(不影响主 P/R/F1)。

## 首个基线(最新代码, preset=balanced, 9个有GT违章的视频)
**总体 P=0.212  R=0.778  F1=0.333  命中段平均覆盖=0.338  车牌=1/7**

| 视频 | P | R | 覆盖 | FP | 备注 |
|---|---|---|---|---|---|
| 02 | 0.20 | 1.0 | 0.37 | 4 | 碎片化 |
| 03 | 0.12 | 1.0 | 0.69 | 7 | 重叠重复事件(见下) |
| 04 | 0.00 | 0.00 | 0.00 | 0 | **漏检**(违章仅42-43.2s) |
| 05 | 0.14 | 1.0 | 0.37 | 6 | 碎片化 |
| 06 | 0.33 | 1.0 | 0.21 | 2 | |
| 07 | 0.20 | 1.0 | 0.43 | 4 | |
| 08 | 0.50 | 1.0 | 0.05 | 1 | 车牌✓(唯一命中) |
| 09 | 0.33 | 1.0 | 0.24 | 2 | |
| 11 | 0.00 | 0.00 | 0.00 | 0 | **漏检** |

## 已确认事实(诊断)
- **灯态不是瓶颈了**(92.9%), 端到端 F1 只 0.33, 瓶颈在**车辆/事件形成 + 车牌关联**。
- **Precision 灾难: 26 FP vs 7 TP**。Recall 0.778 尚可(7/9 违章被 flag), 但每个真违章被切成多短事件 + 非违章静止时刻误报。
- **最高杠杆可修 bug**: `violation_engine` 的 `_dedup`(同 track_id 间隔<gap 合并)**不合并跨 track 的重叠事件**。
  违章03 实证: 输出 `[91.5-110.9]`、`[91.5-97.0]`、`[91.5-92.1]`、`[94.0-109.9]` 等大量**时间重叠**的 confirmed 事件
  -> 见 `src/redlight/pipeline/violation_engine.py:251` `_dedup` 按 track_id 分组, 碎片化的多 track 各自成事件不被折叠。
- **2 漏检**(04/11): 短时违章 + 车辆静止判定未持续满足(与 07-14 e2e handoff 记的车辆碎片化同源)。
- **覆盖率仅 34%**: 即便命中, 事件时长远短于真实违章(如 08 仅 5%、09 仅 24%)。
- **车牌关联几乎全废(1/7)**: 事件 plate 靠 `_write_outputs` 按 track_id 从 PlateConsensus 回填, track 碎片化 -> track_id 对不上 -> plate 空。

## 下一步动作(建议优先级)
1. **降 FP(最高杠杆)**: 在 `violation_engine` 出事件前, 对**时间重叠**的 confirmed 事件做跨 track 合并/NMS
   (现只按 track_id 去重)。预计能把 26 FP 大幅砍掉, Precision 从 0.21 显著上升。跑 `eval_violations.py` 验证。
2. **修漏检 04/11**: 排查静止判定为何在短违章段不持续满足(track 碎片化/IoU静止阈值)。
3. **修车牌关联**: track 合并后 track_id 稳定, plate 回填应自然改善; 或改为按事件时空范围聚合车牌而非纯 track_id。
4. 覆盖率: 事件合并后应一并改善。

## 约束
- eval 跑全量约 6-9 分钟(11.8~8 推理帧/秒, 9视频)。stdout 被缓冲, 跑完才出结果。写 `data/output/eval_violations/`(gitignore)。
- `--reuse` 只对已有 `data/output/run_<video>_<preset>/violations.csv` 有效(当前仅少数视频有)。
- 多 agent 仓库: scoped `git add`, 署名。
