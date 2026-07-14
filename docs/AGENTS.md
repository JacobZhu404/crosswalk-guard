# 多 Agent 协作规范（AGENTS.md）

> 生效：2026-07-13 ｜ 维护人：Jacob Zhu（仓库主 / 合并权限）｜ 适用范围：本仓库所有人类 + Agent 协作者
> 配套：阶段交接用 `docs/handoff/YYYY-MM-DD-<topic>.md`；设计决策用 `docs/plans/`。

---

## 0. 为什么要有这份文档

本项目同时有多个 agent 在同一仓库并行工作（算法开发、测评、文档等）。已发生过两类真实事故，必须用硬规则避免再犯：

- **事故 A（并行 reset 冲突）**：某 agent 在同工作树 `main` 上批量 `git checkout/pull/reset`，差点冲掉另一 agent 未提交的评测产物。
- **事故 B（GT 文件 ACL 锁，E21）**：用**管理员模式**编辑器独占打开 `datasets/gt/*.csv`，关闭后文件 ACL 变成"仅管理员可读写"，普通用户/工具全部 `Permission denied`，修复极耗时（takeown/icacls 都拒，最后靠 SYSTEM 任务才解开）。

> **一句话原则**：每个 agent 只动自己的分支和自己的输出目录；共享文件（GT / 主干）必须协调；所有提交必须署名。

---

## 1. Agent 注册表（协同署名）

新 agent 入项目，**先在此表加一行**；后续提交用对应的 `Co-Authored-By` 署名。

| Agent ID | 显示名 | 职责 | 分支策略 | 输出目录 | 状态 |
|---|---|---|---|---|---|
| `human-jacob` | Jacob Zhu（人类，仓库主 / 合并权限） | 最终裁决、合并到 main | — | — | active |
| `senior-dev` | Senior Developer（高级开发工程师） | 检测算法 / 评测基建 / 文档 | 默认 main；大功能拉 `feat/<topic>` | `data/output/senior-dev/` | active |
| `eval-agent` | 并行测评 Agent | 红绿灯 / 事件级评测 | 默认 main；大功能拉 `feat/<topic>` | `data/output/eval-agent/` | active（确切名待用户确认） |
| `plate-agent` | Plate Recognition Agent（车牌识别 Agent） | 车牌识别优化 / 评测集 / 标注画廊 | 默认 main；大功能拉 `feat/<topic>` | `data/output/plate-agent/` | active |
| _（新 agent 在此追加）_ | | | 默认 main；大功能拉 `feat/<topic>` | `data/output/<id>/` | |

**署名邮箱约定**：`<agent-id>@crosswalk-guard.agents`（虚拟域，仅用于 commit trailer 标识，不收发邮件）。

---

## 2. 分支与输出隔离（防互踩）

- **分支（默认主干开发）**：**日常开发直接在 `main` 上进行**，提交小而聚焦、配套单测一并提交即可，无需每人拉独立分支。**仅当遇到较大的、需要上下联调、持续较久的功能**时，才从 `main` 拉独立分支 `feat/<topic>` 开发，联调稳定后再合回 `main`。
  - 无论是否拉分支，都**禁止对共享树执行 `git reset --hard` / `git checkout .` / `git clean -fd`**（见 §3 红线）与 `git push --force origin main`。
  - 已合并回 main 的临时分支（如早期 `feat/light-eval-v2`）可保留历史，不再作为活跃开发分支。
- **输出目录**：评测 / 运行产物写到 `data/output/<agent-id>/`，**绝不写共享的 `data/output/light_eval/` 等其它 agent 的目录**。
  - 例：并行测评 agent 跑 `eval_light_fast.py` 时，把输出重定向到 `data/output/eval-agent/`，不要覆盖 `data/output/light_eval/`。

---

## 3. 禁止操作（红线）

1. ❌ 在共享工作树执行 `git reset --hard` / `git checkout .` / `git clean -fd` —— 除非你 **100% 拥有**所有未提交改动。
2. ❌ `git push --force origin main`（强制推主干）。
3. ❌ 写 / 覆盖其它 agent 的输出目录或分支。
4. ❌ 用**管理员模式**编辑器独占打开 `datasets/gt/*.csv` 等共享文件（触发 ACL 锁，见 E21）。如必须用管理员编辑器，改完**立即关闭标签页**释放锁。
5. ❌ 并发跑写同一份 GT / 评测文件的脚本。

---

## 4. 共享 GT 文件协议（`datasets/gt/*`）

GT（`events.csv` / `videos.csv` / `light_state/` / `violation_events/`）是**全项目唯一真值**，多人会读会改：

- 改 GT 前先在注册表 / 群里声明你负责哪部分（建议**按视频分区**，如 A 负责 01–05、B 负责 06–11）。
- 优先**追加行 / 提交 diff**，不要整文件重写。
- 改完立即 `git commit` + `git push`（到你的分支或经 maintainer 合入 main），避免别人基于旧版本工作。
- 用**普通权限**编辑器（非管理员）打开，避免 ACL 事故。

---

## 5. 提交与署名规范

- **Conventional Commits**：`type(scope): 中文摘要`。`type ∈ {feat, fix, docs, test, refactor, perf, chore}`。
- **强制 trailer（署名）**：每条 agent 提交必须含
  ```
  Co-Authored-By: <显示名> <<agent-id>@crosswalk-guard.agents>
  ```
  例：`Co-Authored-By: Senior Developer <senior-dev@crosswalk-guard.agents>`
- 人类 Jacob 的提交为自然作者；agent 提交由 Jacob（或触发者）作 author，agent 作 co-author。
- 提交要**小、聚焦**；配套的单测 / 评测改动一同提交。

---

## 6. 合并到主干协议

- `main` 受保护，仅 maintainer（Jacob）合并。
- agent 完成 → push 自己的分支 → 通知 Jacob 合并（GitHub PR 或口头）。
- 合并方式：merge（保留历史）或 squash；合并后分支可删（或归档到 `archive/`）。
- **合并前先 `git pull --rebase` 拿到最新 main**，解决冲突再请求合并。

---

## 7. 交接（Handoff）规范

- 阶段性成果用 `docs/handoff/YYYY-MM-DD-<topic>.md`（已有范式），含：日期 / 阶段 / 负责人(agent) / 任务目标 / 已完成清单 / 阻塞点 / 如何验证 / 下一步。
- 长线经验沉淀进 `src` 注释 + `docs/plans/` 设计文档 + 项目 MEMORY（如有）。

---

## 8. 冲突自查清单（每次提交前）

- [ ] 我在 main 上做日常提交，或（大功能）在自己的 `feat/<topic>` 分支？
- [ ] 输出只写到 `data/output/<我的 id>/`？
- [ ] 没动别人的目录 / 分支？
- [ ] 没对共享树做 `reset` / `checkout .` / `clean`？
- [ ] 提交含 `Co-Authored-By` 署名？
- [ ] 若改了 GT，已声明分区并立即提交？
