# Handoff 交接快照 [LINGMA 全量代码审阅]

> 生成时间: 2026-07-13
> 负责人: Lingma (Mac 开发机)
> 状态: 分析完成，待用户/Claude Code 评审确认后执行

---

## 1. 核心任务目标

- **原始需求**: 读取整个工程（文档+代码），给出分析结论，经用户确认后再动作，避免多 agent 打架。
- **验收标准**: 输出完整问题清单+修复建议，按优先级排序。
- **约束规则**: 不改代码只分析；多 agent 环境需署名；Mac/Windows 双机同步靠 git。

---

## 2. 已确认事实

1. 工程整体架构良好（6 层分层），代码结构清晰，有 pytest 单元/集成测试。
2. Git 工作树干净，最新提交 `ae115d3`，无未提交改动。
3. 语义已反转：违规 = 行人绿灯/闪烁 + 车辆静止 + 压斑马线。
4. 红绿灯检测器当前为 `color-v7-head`，02/03/04 已用 `light_priors.json` 标定。
5. COT 小作文代码已写入，但存在 `import json` 缺失（P0#1）。
6. `compute_overlap_ratio` 引擎端用 `denom="mask"`，可视化端仍用默认 `"box"`（P0#4）。
7. 本机为 macOS/Python 3.7.5，`cv2/torch/ultralytics/hyperlpr3` 均未安装，只能跑纯逻辑测试，主流程需在 Windows 验证。

---

## 3. 已排除项（防重复试错）

- 对比度门控：已从 `_candidates` 移除（handoff 2026-07-12 第 8.3 节实证失败）。
- `frac_v>=0.30` 门控：已废弃（E20 实证误杀真绿灯）。
- 回退到 v6 单轨迹选灯：已废弃（同杆红绿合并后红灯赢）。
- "全场景红/绿像素发射求和"选灯：已排除（场景红会淹没正确绿灯）。

---

## 4. 修改文件清单（待执行，当前未修改任何文件）

| 优先级 | 文件 | 拟修复内容 |
|--------|------|-----------|
| P0 | `src/redlight/app/cli.py` | 补 `import json`；`choices` 加 `"very_loose"`；默认 `mode` 改 `"pedestrian_green"` |
| P0 | `src/redlight/pipeline/visualizer.py` | `compute_overlap_ratio` 显式加 `denom="mask"` 与引擎一致 |
| P0 | `src/redlight/pipeline/dag.py` | `cw.detect()` 传入 `vehicle_boxes` 启用锚定 |
| P0 | `src/redlight/pipeline/violation_engine.py` | 默认 `mode` 改 `"pedestrian_green"` |
| P1 | `configs/config.yaml` | 更新 traffic_light 注释（v6 -> v7-head） |
| P1 | `tests/unit/test_traffic_light.py` | 清理 v6 废弃参数（`lit_emission_floor` 等） |
| P1 | `handoff_plate_optimization.md` | 标记为过期/删除（与语义反转冲突） |
| P1 | `pyproject.toml` | 版本号与 README 对齐（2.0.0 -> 2.3.0） |

---

## 5. 约束条件

- Mac 端只能做纯代码修改和 pytest 逻辑测试，**不能跑含 cv2/torch 的主流程**。
- 任何修改后须 commit 并 push，Windows agent 拉取验证。
- 用户规则：重大修改须 commit；多 agent 冲突由 Claude Code 裁决。

---

## 6. 下一步动作

**等待用户/Claude Code 评审后确认执行范围。**

建议选项：
1. 立即修 P0#1~#5 + P1#6~#9（纯代码 safe fix，风险极低）。
2. 只修 P0（崩溃/判反类问题）。
3. 先跑 mock 测试确认基线，再动手。
4. 用户指定其他优先级。

---

*本快照用于上下文压缩前交接。恢复工作时先读本文件 + `docs/plans/2026-07-12-design-requirements-v2.md`。*
