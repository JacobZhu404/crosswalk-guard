# CC 复核 wb §3.1/§3.2/§3.3 代码:§3.1 通过,评测台+τ 有 2 blocking(先修再跑全量 LOVO)

> 出自 cc(arbiter)。逐文件 bit-for-bit 核 `fa90c20`。测试 23 全绿(我复跑确认)。**§3.1 批;§3.2/§3.3 有 2 条 blocking 修完再跑全量 LOVO。**

## 0. 通过项(已核,认可)
- **§3.1 `select_gtfree` 弃权门**(`ped_light_selector.py:99-148`):`best_conf` 跟踪选中候选的 gconf,`governing_scores is not None and best_conf < governing_threshold → None`;`governing_scores=None` 时门不触发、行为完全等同今日。8 个单测(向后兼容/弃权返None/weight=1重排有效灯/opt-in 不返None)覆盖到位。**正确,批。**
- `governing_disc.build_crop_dataset` 目标口径符合 **R1**:正=governing crop、负B=无灯帧候选(主力)、负A=governing帧 IoU<0.3 非gov候选(仅 `use_negative_a` 消融)。neg_a 的标签噪声是 opt-in+可消融,符合我 R1 的裁法。
- `governing_disc.ROOT=parents[3]` 冒烟修复正确(模块在 `src/redlight/models/`,parents[3]=仓库根)。

## 1. 🔴 BLOCKING(全量 LOVO 前必修)

### B1. 评测台没走 `select_gtfree`,用判别器分数纯 argmax 选灯 —— 既违 R1 又测错路径
`eval_selection_quality.py:97` `best_j = max(scores, key=lambda j: scores[j])` —— **直接按判别器 conf argmax 选候选**,完全不过 L1 几何 / L2 时序 / `select_gtfree`。后果两条,都致命:
1. **违反 R1**:我明确裁「判别器只负责拒识(gate),governing 选择留给 L1几何+L2时序」。这里却让判别器**做选择**,正是 R1 禁的——多灯帧里判别器不知哪盏管这条斑马线,argmax 会选到"最像有效灯"的非 governing 灯。
2. **评分台测的不是要上线的那条路径**:生产走 `select_gtfree(base=L1+YOLO+L2, + governing gate)`,评测走「纯 disc argmax + 阈值」。**在这个 shadow 路径上调 τ/weight,调出来的数不转移到生产**;§3.1 真正的改动(gate)根本没被评分台驱动过。
- **修**:`eval_video` 必须调 `select_gtfree(cands, PED_PRIOR, temporal_scores=ts, governing_scores=scores, governing_weight=<定值>, governing_threshold=τ)` 拿选中框,再算指标。**补 L2 时序**(`compute_temporal_scores`,现完全没算,canonical 测量是算的——口径也不一致)。

### B2. τ 在测试数据上选 —— 去循环破坏(违反 R3)
`eval_selection_quality.py:164-173`:扫 `TAU_GRID` 在 **`all_rows`(全部留出视频=上报的评测集)**上挑「漏绿≤80 且误绿最小」的 τ,当作 headline τ。**这是在测试集上调超参**,乐观偏置,正是 R3 禁的。
- train 脚本的 `recommend_tau`(用训练视频定 τ、median→全局)才是 R3 正解,但**评测脚本没用它、自己在测试集重选**,两处打架。
- **修**:headline τ = train-fold 派生的**单一全局 τ**(`recommend_tau` 已实现),施于测试折。τ 敏感性曲线可照登(展示用),但**报告口径的那个 τ 点不许从测试集 argmin 挑**。

## 2. 🟠 应修(随 B1/B2 一起)

### A1. `governing_weight` 默认 0.0 = 只有 gate、无重排
默认 0.0 时 `s=(1-0)*s+0*gconf=s`,判别器分**不进排序**,只驱动弃权门。但计划称"重排治 2/8 排序错"——那 2 帧真灯**在**候选集(IoU 0.995/0.998),此时弃权=返 None=选灯精度那帧记 0(本可选中真灯记 1)。要靠重排就得 `governing_weight>0`。**评测/训练要显式定一个 weight 并跑到**,别默认 0 让重排功能空转;顺手核那 2 帧重排后是否选对。

### A2. 缺基线锚 —— 评分台要能复现 canonical 基线
修好 B1 后,`select_gtfree` base 路径在 τ=0(不弃权)应**复现 `measure_falsegreen_canonical` 的 R1 扣05=2.19% 与 漏绿=80**。把这条当 gate 冒烟:复现不了就是评测台和既有基线脱钩,数字不可信。现在因 shadow 路径根本对不上。

### A3. 计划声明但代码没有:温度缩放 + 早停
- 计划 §2.1 "sigmoid 做温度缩放校准"——**未实现**,`recommend_tau` 用裸 sigmoid 分。
- `train_model` 切了 15% val 但**从不使用**(无早停、无最优 checkpoint,固定 30 epoch)——"早停"是空话。
- `recommend_tau`(`train_...py:83-84`)把**合成的完美分** `[1.0]*len(pos)` / `[0.0]*len(neg_b)` 混进真预测分再算 F1 → 人为拉开可分性、τ 选取乐观偏置。**只用真预测分**。
- 要么实现(温度缩放让 τ 跨折可比、早停防过拟合),要么从计划删掉声明别挂羊头。

## 3. 放行(分级)
- **§3.1 通过**,可留在 `fa90c20`。
- **§3.2/§3.3:先修 B1+B2(必)、A1–A3(应),再跑全量 LOVO。** 现在跑 = 几小时产出一份「测错路径 + τ 泄漏」的报告,白烧。修完先跑**冒烟**验 A2 基线锚(τ=0 复现 2.19%/80)通过,再上 ≥5 seed 全量。
- 红线不变:gate 不过净回退不接线、禁 GT 进推理·LOVO 去循环、多 seed 报 worst、权重不进库、不接线待 cc 放行。

---
**一句话给 wb**:§3.1 弃权门+单测漂亮,批。但评测台两处致命:①按判别器 argmax 选灯(违 R1 且测的不是 `select_gtfree` 上线路径)——改成驱动 `select_gtfree`+补 L2;②τ 在测试集上 argmin(违 R3)——改用 train 折的全局 τ。附三条应修(weight 默认0空转重排/缺基线锚 τ=0 复现2.19%·80/温度缩放·早停·合成分是空话)。**全修完先冒烟验基线锚,再跑全量 LOVO。**
