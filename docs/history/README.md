# docs/history — 历史文档归档

> 本目录存放**已被后续决策取代**的 handoff / plans / reports / diagnostics。
> **Agent 日常工作请忽略本目录**;`docs/` 顶层只保留当前有效的工作集。
> 归档=去噪,**非删除**:内容完整保留在库里(git 历史亦在),需回溯"当年为什么这么定"时来这里查。
> 注:命名用 `history/` 而非 `archive/` —— 后者被 `.gitignore` 约定为"仅本地、不入库",本目录要**入库保留**。

归档时间:2026-07-26(架构转向"逐帧检测"后的一次集中清理)。

## 当前有效文档(未归档,在 `docs/` 顶层)
- `docs/AGENTS.md` — 多 agent 协作规范(始终有效)
- `docs/plans/2026-07-12-design-requirements-v2.md` — **权威 spec**(违章语义/系统设计 v2)
- `docs/plans/2026-07-12-gt-format-spec.md` — GT 格式规范
- `docs/plans/2026-07-22-wb-plan-perframe-detection-remine-classify.md` — 当前架构实现计划(逐帧检测)
- `docs/handoff/2026-07-2[1-3]-*` — 转向逐帧检测的根因/诊断/审计决策链(prior 偏框→手持运动→YOLO 可用→架构锁定→计划审→弱点审计→L3 verify)
- `docs/diagnostics/2026-07-21-wb-diag{1,2,3-v2-corrected}.md` — 支撑当前架构的诊断(diag3 取 v2 修正版)
- `docs/reports/2026-07-25-wb-l3-selector-gate.md` — L3 选灯器最新 gate 报告
- `docs/diag/model-dataset-survey-2026-07-12.md` — 外部模型/开源数据集调研(fork A 公开数据侦察仍会用到)

## 归档内容分类
- `handoff/` — 07-12~07-21 的阶段交接(phase1/e2e/crosswalk-flicker/vehicle-anchor/false-green/phaseA/phaseB 等,均被逐帧检测转向取代)
- `plans/` — 被取代的设计与实现计划(v3~v8 车辆锚定/VLM 标注/斑马线检测升级/phaseA·B 重训/step2 信号过滤 等)
- `reports/` — 对应的 wb 诊断/结果报告
- `misc/` — diag3 未修正版(被 v2 取代)、07-12 信号类型诊断、灯态标注诊断指引、根级两个车牌任务旧快照
