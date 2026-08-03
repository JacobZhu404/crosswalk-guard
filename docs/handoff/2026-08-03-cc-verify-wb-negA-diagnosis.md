# CC 复核 wb neg_A 诊断(c888919)— 污染闸门通过, 但揪出 1:14 类不均(confound+漏绿gate风险)→ 全量前先控比例

> 出自 cc(arbiter)。wb 交 A 消融前置诊断(分支 `wb-negA`, commit c888919)。cc 独立复核: 代码隔离 + 污染闸门亲验, 并揪出 wb 诊断未标的训练不均问题。

## 0. 通过项(cc 亲验)
- **代码隔离干净**: `--use-negative-a` flag 线穿 `_load_or_train`/`_tau_for_fold`/`_eval_cached`/`_global_tau`, 缓存/报告全走 `*_A`/`*_s1A`/`-negA` 新路径, **不碰 B-only s1 FAIL 锚**; 且顺手补了 `_atomic_write`(tmp+os.replace, 合红线)。wb 在独立 worktree `crosswalk-guard-wb/` 作业, 隔离合规。
- **污染闸门通过(cc 亲看 4 张最高风险 ctx, 非采信)**: #01=信号灯头**外壳/背面**(本帧真亮的是下方**红**灯, 未框, 非绿)、#09=公交车内人影(0.15 IoU 偶碰)、#11=灯杆(帧顶确有真绿灯但**未框**)、#14=水果纹布料。全为干扰源, **0 张平行斑马线真绿灯**。wb 的 max_iou_gov=0 不排除真灯、必须看全帧 ctx 的判断**方法正确**(是我 spec 要的那种审)。20/20 污染率 0% 采信。

## 1. ⚠️ 揪出问题: neg_a 引入 1:14 类不均(wb 诊断未标)
- 逐折: pos≈650, neg_b≈1000, **neg_a 6.6k–9.4k(占负样本 85–90%)** → 总 pos:neg ≈ **1:12–1:16**。B-only 是 pos:neg_b ≈ 650:1000 = **1:1.5**。
- `train_model` 用 `nn.BCEWithLogitsLoss()` **无 pos_weight / 无类平衡 / 无采样重加权**(`governing_disc.py:219`)。1:14 下负样本主导梯度 → 决策边界压向"判负" → 更多候选跌破 τ 弃权。
- **两重危害**:
  1. **漏绿 gate 风险**: B-only worst-seed 漏绿已 **=81(>80 边界)**。这个偏斜大概率把漏绿**推得更高**, 不是更低 → 全量很可能因**不均伪影**判 FAIL, 而非 neg_a 本身没用。
  2. **confound(混淆消融)**: R1 "带A vs 不带A" 本意隔离"同帧 lit 干扰"这**一个**变量; 但 1:1.5→1:14 同时改了类比例。若全量回退, 无法归因是"neg_a 内容"还是"负样本压倒性偏斜"。
- 温度缩放/`recommend_tau` 也在这个偏斜分布上定 T 与 τ(F1 代理), 一并被带偏。

## 2. 裁定 — 污染过关, 但**全量 5-seed 暂缓**, 先控负样本比例
- neg_a **内容合格**(方向对: qw 天花板诊断 [[selection-precision-ranking-bottleneck]] 也指排序有 +39.7pp headroom, A 消融值得做)。
- **但别用 1:14 裸铺全量**——先控比例, 二选一(wb 定并在报告写明):
  - **(推荐)按比例下采样 neg_a**: 每折随机(**seed 决定, 可复现/可缓存**)采 neg_a 使总 neg ≈ **2–3× pos**(~1300–2000), 与 B-only 同量级 → A-vs-B 变成**单变量**(只换负样本"成分"不换"数量")=干净消融, 且消除偏斜风险;
  - **或 pos_weight**: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` 用全量 neg_a 但平衡梯度(注意: B-only 无 pos_weight, 这会让损失函数与 B 也不同, 消融不如下采样干净)。
- 无论哪条, **建议先跑 1 个 seed 看 worst 漏绿**(便宜), 确认不均已控、漏绿没被推爆, 再铺 5-seed。省算力、防白跑。
- 红线不变: gate(worst 漏绿≤80 且 误绿≤base 2.19%)不过不接线; LOVO 去循环; 权重不进库; 报告 worst-seed; cc bit-for-bit 复核全量结果。

---
**一句话**: wb 的 neg_a 污染闸门 cc 亲验通过(4 张最高风险 ctx 全是干扰、0 真绿灯), 代码隔离/原子写合规。**但揪出 neg_a 占负样本 85–90%=pos:neg≈1:14, BCE 无 pos_weight → 压向弃权 → 漏绿(已在 81 边界)大概率被推爆 + 消融被 confound。全量 5-seed 暂缓: 先把负样本控到 ~1:2–1:3(推荐 seeded 下采样 neg_a, 单变量干净消融), 跑 1 seed 验漏绿没爆, 再铺全量。**
