# Handoff — 端到端违章:剩余工作接力 (2026-07-16)

> 署名: Claude Code (Opus) · 交接给后续模型继续推进 · 配套: 2026-07-16-e2e-violation-eval.md(工具/基线)

## 目标(一句话)
把端到端违章 F1 从当前 **0.636** 继续往上推: 修 2 个漏检(04/11)、砍残余 6 个 FP、提车牌命中/覆盖率。
唯一客观标尺 = `python scripts/eval_violations.py`(见下)。**每改一处都要重跑 eval 验证, 不回退。**

## 已确认事实(本轮已完成, 已提交)
- 灯态时序融合 92.9%(commit dcc8dc6); 端到端评测工具 eval-e2e 落地(867edb2)。
- **事件跨 track 全局合并**(e6bdbc9): `violation_engine._dedup` 已从 per-track 改为全局时序合并 ->
  FP 26→6, F1 0.333→0.636, Recall 0.778 不变。episode 带 `member_tracks`, 代表 track=max_overlap。
- **车牌回填遍历 member_tracks**(3f27eaa): `cli._episode_plate`, 车牌命中 1/7→2/7。
- 全 unit+integration 测试绿(见 `python -m pytest tests/unit tests/integration -q`)。

## 当前基线(最新代码, preset=balanced, 9 个有 GT 违章视频)
**总体 P=0.538  R=0.778  F1=0.636  覆盖=0.426  车牌=2/7**

| 视频 | P | R | 覆盖 | FP | 状态 |
|---|---|---|---|---|---|
| 02 | 0.50 | 1.0 | 0.49 | 1 | TP✓ 1个窗外FP |
| 03 | 1.00 | 1.0 | 0.71 | 0 | ✅完美 |
| 04 | 0.00 | **0.00** | 0.00 | 0 | **漏检**(根因见下) |
| 05 | 0.33 | 1.0 | 0.53 | 2 | |
| 06 | 1.00 | 1.0 | 0.42 | 0 | ✅ |
| 07 | 0.33 | 1.0 | 0.50 | 2 | 车牌✓ |
| 08 | 1.00 | 1.0 | 0.07 | 0 | ✅ 覆盖极低 |
| 09 | 0.50 | 1.0 | 0.26 | 1 | |
| 11 | 0.00 | **0.00** | 0.00 | 0 | **漏检**(根因见下) |

## 关键诊断(已用 `--cot` 导出中间态, 省却重复排查)
复现命令: `PYTHONPATH=src python -m redlight.app.cli --video input_video/违章11.mp4 --output /tmp/d11 --cot`
然后看 `/tmp/d11/cot/analysis_违章11.json` 的 light_segments 与 tracks[*].occupancy.max_overlap。

### 漏检 04 (GT 违章 42-43.2s, 仅 1.2s)
- light_segments = `unknown[0-3] / red[3.1-43.1]` — **全程无 green 段**, GT 的 42-43.2s 短绿被完全漏判。
- 78 个 track **全部 max_overlap=0.00**。
- **根因(双重)**: ①灯态漏判短绿(主因, 灯不绿则永不 confirmed); ②占道恒为 0(见共性)。
- 建议: 04 是极短违章(1.2s)硬例, 优先级可低于 11; 若攻则先看灯态为何漏 42s 的绿(可能 observe/YOLO 在片尾丢帧或先验 ROI 偏)。

### 漏检 11 (GT 违章 15-28.4s green, 13.4s)
- light_segments = `unknown[0-5.1] / red[5.2-20] / green[20.1-28.3](conf0.64)` — 绿灯**迟到 ~5s**(GT 15 起, 实测 20.1 起), 但确有绿段。
- 48 个 track **全部 max_overlap=0.00**。
- **根因(主)**: 占道恒为 0 → 即便绿灯段也无车压线 → 无事件。次因: 绿灯迟 5s(缩短了可命中窗)。

### 共性根因(最高优先级!): 04/11 所有 track 占道 max_overlap 恒为 0
- 强烈指向**斑马线掩膜在 04/11 为空/错位**, 或 footprint 重叠计算在这两个视频失效。
- 对比: 02/03/05/06/07/08/09 都有正常 occupancy(能出 TP), 说明是 04/11 特定的掩膜泛化失败(E8 掩膜质量评测一直"计划中"未做, 正是此坑)。
- **验证步骤(便宜模型先做这个)**:
  1. 确认是 ENGINE 路径也为 0(不只 COT accumulator): 在 `scripts/eval_violations.py` 或临时脚本里, 对 04/11 打印 `decide()` 前每个 track 的 `occupancy_intervals` 和 `stationary_intervals`。
  2. 若确为 0: dump 04/11 的斑马线掩膜(`CrosswalkDetector` 输出)存图, 肉眼看掩膜是否为空/位置错。相关 `src/redlight/models/crosswalk.py`(v11 多位置条带扫描+车辆锚定)。
  3. 若掩膜正常但 overlap=0: 查 `infrastructure/geometry.py::compute_overlap_ratio(box, mask, footprint=0.5, denom="mask")` 在 04/11 分辨率/坐标下的行为。

## 已排除项(不要重走)
- **别削弱 eval 口径**迎合数字: 事件级 1:1 重叠匹配是用户选定标准(FP 必须如实计)。
- **别在 `_dedup` 里按状态豁免或改回 per-track**: 全局时序合并是正解, 已验证。
- **灯态别再动 enforce_transition_limit 的吸收逻辑**(dcc8dc6 已定稿): 豁免 unknown/flashing 会让灯态崩到 68%(已实证)。
- 车牌剩余漏牌(5/7)**不是关联问题**(已修), 是 OCR 识别错误(如 02 京JL1300 vs GT 京LNE560) → 属车牌 OCR 准确率, 另案。

## 修改文件清单(本轮)
- `src/redlight/pipeline/intermediate_state.py`, `temporal_fusion.py`: enforce_transition_limit(灯态).
- `src/redlight/pipeline/violation_engine.py`: `_dedup` 全局合并 + `_new_episode/_absorb` + member_tracks.
- `src/redlight/app/cli.py`: `_episode_plate` 遍历 member_tracks 回填车牌.
- `src/redlight/evaluation/violation_eval.py` + `scripts/eval_violations.py`: eval-e2e 工具.
- 测试: `tests/unit/test_violation_eval.py`, `test_cli_plate.py`, `test_batch_violation_engine.py`(+跨track合并/dedup直接测).

## 约束条件
- 多 agent 仓库: **scoped `git add <file>` 勿 `git add -A`**; commit message 末尾署名 `Co-Authored-By`。
- 改动先写/跑测试(TDD), 再跑 `pytest tests/unit tests/integration -q` 全绿, 再跑 `eval_violations.py` 验证端到端不回退。
- eval 跑全量 ~6-9 分钟; **stdout 被缓冲, 进程结束才出结果**(用 `> file 2>&1` 后等完再看)。产物写 `data/output/eval_violations/`(gitignore)。
- `--reuse` 读已有 `data/output/run_<video>_<preset>/violations.csv`(可能是旧结果, 慎用; 改代码后必须 fresh 跑)。

## 下一步动作(唯一首要)
**验证并修复 04/11 的占道 max_overlap 恒为 0**: 先按上面「验证步骤 1」确认 ENGINE 路径 occupancy 是否为 0;
若是 → 「步骤 2」dump 掩膜定位是斑马线检测(crosswalk.py)还是重叠计算(geometry.py)的问题。这是同时解 2 个漏检 + 提覆盖率的最高杠杆点。修完跑 `eval_violations.py` 看 R/覆盖是否上升。
