# cc 效果 gate:dedup 修复(wb `45348b6`)全 11 视频 = PASS

> 署名:cc(arbiter) 转交:Jacob;抄送 wb
> gate 对象:wb `45348b6`(worktree `wb-dedup-review-fix`,Plan A **标签版**:`review_reason`)
> 环境:cc 独立 detached worktree @`45348b6`(非 wb 活动 worktree,红线),symlink 本地 `input_video`/`models`,生产 v2/box 全 11 视频含负例,`eval_violations.py --detector v2 --occ-denom box --min-overlap 0.5`。

---

## 1. gate 结果(真实管线,cc 亲跑)

**总体:P=1.000 / R=0.889 / F1=0.941(tp8 / fp0 / fn1),真误报=0,碎片=0。**

| 视频 | P/R/F1 | confirmed 段 | GT | 判定 |
|---|---|---|---|---|
| 01[负例] | —/—/— | (无) / 1 待复核 | (无) | **0 FP** ✓ |
| 02 | 1.00 | [23.6-74.2] | [21-68] cov0.94 | TP |
| 03 | 1.00 | [91.5-137.8] | [74-134] cov0.71 | TP |
| 04 | 0/0/0 | (无) | [42-43] | **fn1 老灯态硬难例,不变** |
| 05 | 1.00 | [0.7-64.5] | [0-30] cov0.98 | TP(脏袋巨块仍在=b2 线) |
| 06 | 1.00 | [0.7-42.0] | [0-29] cov0.98 | TP |
| 07 | 1.00 | [4.2-49.4] | [0-43] cov0.90 | TP |
| 08 | 1.00 | [0.7-49.2] | [0-51] cov0.95 | TP |
| **09** | **1.00** | **[0.7-106.3]** | **[11-72] cov1.00** | **TP 恢复**(车牌 1/1)✓ |
| 10[负例] | —/—/— | (无) | (无) | **0 FP** ✓ |
| 11 | 1.00 | [18.0-28.3] | [15-28] cov0.77 | TP |

**验收锚逐项达成**:09 confirmed 命中[11-72] ✓ / 01·10 仍 0FP ✓ / 02·03·05·06·07·08·11 零回退 ✓ / 04 fn1 维持 ✓ / F1≥0.941·P=1.000 ✓。对比合 #3 前 0.889、现状(未修 dedup)0.875 —— **鱼与熊掌兼得**。

## 2. gate 结论

**wb `45348b6` 的 dedup 修复效果 gate = PASS。** 修复正确解开 09 的 review 毒化(#3 瞬态绿不再拖垮 55.9s 铁绿上的 confirmed 核),且不复活 01 负例、不回退任何其他视频。

## 3. 待 Jacob 拍板:代码形态(与效果无关,两版结果一致)

**分歧**:Jacob 上轮拍「简单版」(2 行全局反转,`c0d4e1d`);wb 依旧 brief `4329e78`(那里「方案A」=标签版)交付了**标签版**(`review_reason` 打标,decision.py+violation_engine.py+6 单测),未见 c0d4e1d 锁。**两版在 11 视频上结果完全一致**(gate 已证;无 D1 遮挡+confirmed 共存 case),标签版对未来该类 case 更安全且已完成测试。cc 倾向**保留 wb 标签版**(已做完、更稳、gate 已过),但此为 Jacob 明确指令的反转点,交 Jacob 定。

---
*署名:cc(arbiter)。证据=/tmp/gate_dedup_out.txt(本文表格即其摘要)。承 [[postmerge-f1-875-09-regression]]、cc 裁定 `cf2e94e`。gate 在 detached worktree @45348b6 独立复跑,未改 wb worktree。*
