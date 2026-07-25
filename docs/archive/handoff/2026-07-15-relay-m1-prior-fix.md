# Handoff — 接力 senior-dev: ③修03prior + ②02漂移诊断 + ①M1评测 (2026-07-15)

> 署名: Claude Code (协调者/接力者) · 日期: 2026-07-15 · 接力 senior-dev handoff §4 待办
> 前序: senior-dev 在 Windows 做了 M1训练脚本+时序修复+01/05/06/07先验+回归集(11提交), handoff §5 M1评测数字空着未跑

## 接力顺序: ③(修03prior) → ②(02漂移) → ①(M1评测)

## ③ 修违章03 prior — ✅ 完成 (commit 472d9a1)

**问题**: 03 prior (0.77,0.05) 在画面顶部边缘, 对到反射绿, 红灯段0-14s整段报绿(289错, acc 57.2%)。

**诊断**: 用 identify_pedestrian_signal.py 数据驱动重识别 → 建议 (0.6,0.5) score=1.0。位置反直觉(中下方)但 03 是反光难例, 信号灯反光在该位置。

**验证** (eval_temporal_fusion):
| 视频 | 改前 | 改后 |
|---|---|---|
| 02 | 82.1% | 82.1% (持平) |
| 03 | 57.2% | **71.9%** (+14.7%) |
| 04 | 84.8% | 97.3% (+12.5%) |
| 总体 | 69.8% | **~80%** |

mismatch 03: 289→190。**印证今日诊断: 03根因是prior位置非HSV阈值。**

## ② 02 t=76-85s ROI漂移 — ⚠️ 结构性, 超出调prior范围

**问题**: 02 绿灯段76-85s, prior (0.72,0.17) 附近无绿色亮斑, 信号灯漂移到远处(0.41~0.99 cx), observe判red(GT=green)。

**诊断**:
- 绿灯段绿色亮斑聚集在 (0.1-0.2, 0.1-0.3) 左上(27+12帧), 非当前prior(0.72,0.17)
- 但 identify 建议 (1.0,0.05) 测了 → acc 暴跌12.4%(那位置是干扰源, 不是信号灯)
- 当前 prior (0.72,0.17) acc 82.1% 已是最优, 76-85s漂移是手持大幅晃动致信号灯整画面移动

**结论**: 单 prior 抓不住手持晃动的信号灯漂移。需**第二先验点 / 信号动态跟踪** (架构改动, 非调参)。记为已知结构性缺口, 留独立任务。

## ① M1评测填数字 — ✅ 完成 (填 senior-dev handoff §5)

senior-dev handoff §5 结果数字空着(没跑完)。接力跑完:

**LOVO 泛化** (弱标签+平衡采样, `--no-verified-only --balanced --epochs 40`):
- 平均 test_acc = **0.484** (02=43.4% / 03=45.6% / 04=56.2%) — 泛化差

**端到端灯态对比** (eval_m1_light.py, detect 路径):
| 视频 | color acc | ped_classifier acc |
|---|---|---|
| 03 | 0.447 | 0.610 ↑ |
| 04 | 0.953 | 0.535 ↓ |

**结论: ped_classifier 整体不如 color, 弱标签 M1 不可用**
- 03 上 ped_classifier 优于 color, 但都低于 eval_temporal_fusion 的 71.9%(detect 路径不如 observe→fuse)
- 04 上 ped_classifier 远差于 color
- green 类 ped_classifier precision/recall 全 0 — M1 几乎认不出绿灯, 把绿灯判成 red/off

**根因** (印证 senior-dev §4 R1/R3 风险):
- 弱标签 crop 质量差: 03 prior 之前错(0.77,0.05), crop 抽错位置; 虽③已修prior但crop是旧prior抽的
- green 正样本太少: 03 walk 232张, 04 walk 仅5张 → M1 没学到绿灯特征
- 必须人工校验 crop verified=1 才能训出可信 M1 (R3)

**模型**: `models/ped_signal.pt` 已导出(gitignore, 不入库; 弱标签版, 不可用)。

## 协作状态理顺 (commit 82b8ad7)
- pull 拉到 senior-dev 11提交; labels.csv 入库策略冲突(我ignore整目录 vs senior-dev期望同步)
- 裁决: labels.csv 入库(我重抽2656正负1:1版覆盖), crop jpg继续ignore
- .gitignore: datasets/ped_signal/*.jpg + */*.jpg; 删冗余行

## 下一步
1. **M1 要可用, 必须先人工校验 crop verified=1** — 用今天建的灯态画廊(make_light_gallery)标, 或基于③修后的新 prior 重抽 crop 再校验。弱标签 M1(acc 48%, green 全错)不可用。
2. **重抽 crop** — 03 prior 已修(0.6,0.5), 但当前 crop 是旧 prior 抽的; 用新 prior 重抽 build_ped_signal_crops 可改善 03 crop 质量。
3. 02 漂移(任务②)留作架构改动独立任务(第二先验/信号跟踪)。
4. 当前生产用 color+prior(③修后总体~80%)已优于弱标签 M1, method 保持 color。
