# Phase A 灯态判别器重训数据集 — 挖掘完成报告

> 对应计划: `docs/plans/2026-07-20-wb-plan-phaseA-classifier-retrain-dataset.md` (cc 238b2e3 批准, 3 处补修 + 4 决策已落)
> 本文交付 cc review: **split 是否无泄漏 / 类平衡是否合理 / 挖掘启发式是否可靠**。
> 状态: 数据已挖, TDD 全过, 画廊已生成待 Jacob 抽检。**未接线、未训练、未改生产代码。**

## 1. 产物清单

| 文件 | 说明 |
|---|---|
| `datasets/classifier_retrain/labels.csv` | 5745 行弱标签 (crop_path,video,frame_ts,x1,y1,x2,y2,source,label,verified) |
| `datasets/classifier_retrain/manifest.json` | split / 每视频计数 / impostor 来源 |
| `datasets/classifier_retrain/<video>/*.jpg` | 抠图 (gitignored, 可再生成) |
| `scripts/mine_classifier_retrain.py` | 挖掘脚本 (半自动, 复用 observe/crop_box) |
| `scripts/make_classifier_retrain_gallery.py` | 分层抽样抽检画廊 |
| `scripts/apply_classifier_retrain_feedback.py` | 反馈合并 (改标 + verified=1) |
| `tests/test_mine_classifier_retrain.py` | 5 项 TDD (全过) |
| `data/output/classifier_retrain_gallery/gallery.html` | **Jacob 抽检入口** (自包含 base64, 免服务) |
| `datasets/ped_signal_legacy_238b/` | 旧坏数据集归档 (留档, 不合并) |

## 2. 数据集总览

**全局类平衡 (目标 off∈[0.2,0.6]):** walk=2186 / stand=1424 / off=2135 → off 占比 **37.2%** ✓

**按视频 split (无泄漏, 整视频归 train 或 val):**

| video | split | walk | stand | off | impostor | outside | bg | 备注 |
|---|---|---|---|---|---|---|---|---|
| 01 | val | 0 | 221 | 266 | 66 | 183 | 17 | 负例, walk=0 正确 |
| 02 | train | 272 | 197 | 219 | 35 | 162 | 22 | |
| 03 | train | 412 | 412 | 248 | 55 | 158 | 35 | |
| 04 | train | 32 | 177 | 178 | 7 | 161 | 10 | 暗短绿, walk 小但>0 |
| 05 | train | 192 | 0 | 188 | 151 | 28 | 9 | 无 red GT(stand=0 正确) |
| 06 | train | 266 | 81 | 227 | 58 | 161 | 8 | 暗绿(训练学) |
| 07 | val | 236 | 183 | 233 | 19 | 197 | 17 | 暗绿(泛化试金石) |
| 08 | train | 216 | 0 | 10 | 0 | 0 | 10 | 干净视频, 无自然 impostor |
| 09 | train | 433 | 30 | 370 | 174 | 176 | 20 | |
| 10 | train | 0 | 66 | 92 | 15 | 74 | 3 | 负例, walk=0 正确 |
| 11 | val | 127 | 57 | 104 | 4 | 98 | 2 | 不同相机(泛化) |

**impostor 来源分解:** false_green_scan(引擎读绿但 GT 非绿)=584 / outside_prior_green(信号外绿斑)=1398 / prior_off(背景正则)=153。

## 3. cc 三处补修 + 四决策落地核对

- **A. 最终接线模型须全 11 重训**: 计划 §3.1 已写死; 本脚本只挖数据 + manifest, 训练期在 Phase B/C 执行。✓
- **B. TDD「walk>0」豁免负例**: `test_class_balance_exempt_neg` 对 01/10 要求 walk==0; 并新增「stand>0 仅要求 GT 有 red confirmed 段的视频」豁免 05/08。✓
- **C. 全新构建不混旧集**: 旧 `datasets/ped_signal` 归档 `legacy_238b`(不读不混); 新 off=impostor 语义, 全重新挖。✓
- **决策 1-4**: 标签复用 walk/stand/off ✓ / val={01,07,11} ✓ / 抽检采样(impostor+暗绿+04+段边界全核 + 其余20%) ✓ / 旧集归档 ✓。

## 4. 执行中发现的 5 个 bug (均已修, TDD 回归绿)

1. **walk/stand 全为 0 (致命)**: 误用 `gt_lookup.expand_light_evidence`, 它把 `light_states.csv` 第 4 字段 `confidence` 当 `evidence` 解释 → `visible` 永不命中。改为专用 `_visible_state_at`(基于 `confidence=="confirmed"`)。
2. **impostor_outside 爆炸 41480 张**: 信号外绿斑第三源无上限。改为预算截断 `max(200, impostor×1.5)` + 统一 off 预算 `1.5×(walk+stand)`。
3. **impostor 误标真绿帧**: outside 源在真绿帧(prior 外恰有绿斑, 如 02 t=31.35)也触发。加 `not(真绿可见)` 守卫, TDD#3 守住。
4. **05 类平衡超标 (off 0.73)**: 05 无 red GT→stand=0→walk 过采样被 `min(stand,...)` 卡死。改为 off 预算截断 + walk 过采样参考 `max(stand, off)`。
5. **画廊反馈路径不匹配 + 缺 import**: feedback 用绝对路径而 labels.csv 存相对 → `apply` 零匹配; 已让画廊存相对路径 + 补 `import base64`。

## 5. 启发式可靠性评估 (供 cc review)

- **真信号 (walk/stand)**: GT `confirmed` green/red 段 + prior ROI 直抠。06/07 暗绿、04 短绿自然覆盖 (TDD#4 硬覆盖过)。04 walk=32 (过采样自 ~12 原始帧) — 刀尖例, 量小但存在。
- **impostor (false_green_scan=584)**: 引擎 `observe()` 读绿且非 GT 真绿可见帧 → 真·误绿 (01 背心绿、483 帧环境绿等)。**最珍贵**, 预算截断时最后才砍。
- **impostor (outside=1398)**: prior 外 HSV 强绿斑 → 真·非信号绿。守卫保证不落真绿帧。
- **bg (153)**: prior 外随机块, 限量 ≤ off 预算, 防「暗=off」偏见。
- **风险点 (诚实)**: 08 是干净视频, 引擎从无误绿、无信号外绿斑 → 仅 10 个 bg, off 占比 4.4%。强行拉到 0.2 需注入假 impostor(不诚实), 故 TDD 对「无自然 impostor 供给」的视频豁免每视频占比门槛, 仅守全局 37.2%。**请 cc 确认此豁免可接受**。

## 6. 下一步

1. **Jacob 抽检**: 打开 `data/output/classifier_retrain_gallery/gallery.html` (已分层抽 3359/5745 张), 逐张校验弱标签 → 导出 `classifier_retrain_feedback.json` → 跑 `apply_classifier_retrain_feedback.py` (改标 + verified=1)。
2. **cc review**: 核 split 无泄漏 / 类平衡 / 启发式可靠性 → 放行 Phase B (训练 + val 留出门控)。
3. **Phase B gate**: 在 val={01,07,11} 上验证判别器泛化 (尤其 07 暗绿、11 不同相机); 过 gate 后按 A 全 11 重训再接线 (计划 §3.1)。

> 注: 04 仍可能因其融合 (prior_flip 压掉末段绿) 而非分类器过不了 — 按钢锭接受继续漏, 不为它倒转顺序。
