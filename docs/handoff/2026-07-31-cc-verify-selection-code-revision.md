# CC 复核 wb 修订版(7d201a0):2 blocking + 3 应修全部落地验证通过,放行全量 LOVO(附 1 应修 S1)

> 出自 cc(arbiter)。**不取信 relay,逐文件核 `7d201a0` + 亲自复跑锚/测试。**

## 0. 独立验证结果(全部我亲跑,非采信 wb)
| 项 | 我核到的落实 | 判定 |
|---|---|---|
| **B1** 评测台走 select_gtfree | `eval_video`(`eval_selection_quality.py:79-142`)三趟:①建候选(YOLO cls9+HSV) ②`compute_temporal_scores` 补 L2 ③`select_gtfree(cands, PED_PRIOR, temporal_scores, governing_scores, governing_weight, governing_threshold=τ)`。判别器**只喂 governing_scores 做 gate(+可选重排)**,选灯 base=L1+YOLO+0.4·L2 —— **R1 守住**(选灯不压给判别器)。 | ✅ 通过 |
| **B2** τ 来自训练折 | `_global_tau`(`:178-185`)每折 `recommend_tau(models[V], gt, train_videos=去V)` → median 全局 τ;敏感性曲线降级为**纯展示**,headline 用 global_tau,不再测试集 argmin。**测试视频 V 从不参与 τ 选取**。 | ✅ 通过 |
| **A1** weight 不空转 | `--governing-weight` 默认 **0.3(>0)**;`--anchor` 强制 0 复现基线。 | ✅ 通过 |
| **A2** 基线锚 | 分母修为 `had_cands or gt_green`(与 canonical 计数一致:候选帧+无候选真绿帧)。**我亲跑 `--anchor`**: 扣05误绿 **8/365=2.19%**、漏绿 **80** —— 与 canonical 逐字吻合,PASS。 | ✅ **独立复现** |
| **A3** 温度缩放/早停/去合成分 | `forward_logits` 分离 + `forward=sigmoid(logit/T)`;`fit_temperature`(val grid 选最小 BCE,val 来自训练折不碰测试);`train_model` 早停(patience=10 保最优 ckpt);`recommend_tau` 删 `[1.0]/[0.0]` 合成分,只用真预测分。 | ✅ 通过 |
| 单测 | selector 15 + eval 8 = **23 全绿**(我复跑)。eval 单测助手补 `had_cands` 与新指标一致。 | ✅ |

**结论:2 blocking(B1/B2)+ 3 应修(A1–A3)全部落地,且关键的 A2 锚由我独立复现。修订质量达标。**

## 1. 🟠 S1 应修(便宜、跑全量前修 —— 不阻塞 headline 正确性,但曲线现在误导)
`eval_video` 内部就在 `governing_threshold=global_tau` 处**当场弃权**:被弃帧 row 记 `best_cand=None`。而 `main` 的 τ 敏感性扫描又对同一批 rows 用不同 τ 调 `metrics_for_tau`——**τ < global_tau 的点无法"复活"已弃权帧** → **敏感性曲线在 τ<global_tau 段是错的**(headline 在 global_tau 处正确,因生成τ=评测τ一致)。
- **修**:`eval_video` 传 `governing_threshold=0.0`(内部不弃权)、照记 `best_cand`+`best_conf`;弃权门交 `metrics_for_tau` 按每个扫描 τ 施加(它已按 `best_conf>=tau` 判)。选灯赢家只由 weight 决定、与 gate τ 无关,故 headline 不变、曲线全段变正确。
- 便宜(几行),且曲线是讨论 τ 的主要物料,跑几小时全量前先修,免得出一份半段失真的曲线。

## 2. 🔵 S2 次要 caveat(记在报告即可,不修)
`recommend_tau` 在**训练折 in-sample** 重打分选 τ(模型见过这些 crop → τ 略乐观);且用 crop 级 pos/neg **F1 作代理**,非下游"漏绿≤80&误绿最小"目标。R3 核心(测试视频不碰)已满足,故不阻塞;报告注明"τ 为训练折 in-sample F1 选取的代理值"即可,便于解读 LOVO 泛化数字时打折。

## 3. 放行
- **修订版通过。放行 ≥5 seed 全量 LOVO**,前置只加 **S1**(便宜、让曲线诚实)。S2 报告注明即可。
- 全量报告须含:全399帧选灯精度 + 正确弃权率 + 扣05误绿 + 漏绿(≤80 硬门)+ **11视频分解 mean±std + worst-video + worst-seed(min)** + 03单列/06·11 N/A + τ 敏感性(S1修后)。
- 红线不变:gate 不过净回退不接线、GT 不进推理、LOVO 去循环、权重不进库、报告交 cc 复核 + Jacob 抽检后才谈接线。

---
**一句话给 wb**:修订全过、A2 锚我亲手复现 2.19%/80,放行全量 LOVO。跑前顺手修 S1(`eval_video` 传 `governing_threshold=0`,弃权交 `metrics_for_tau` 施加——否则敏感性曲线 τ<global_τ 段失真;headline 不受影响)。S2(τ 训练折 in-sample F1 代理)报告里注明即可。全量出 mean±std+worst-video+worst-seed。
