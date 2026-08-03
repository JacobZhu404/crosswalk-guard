# CC 复核 qw 违章05 候选生成诊断(d6f2eb3)— 数据 bit-for-bit 通过, 但裁定归因错(stage 判定顺序 bug 把 B 掩成 C);实为两个独立 bug:低饱和过滤(天花板主因)+ prior 偏框(生产假绿源)

> 出自 cc(arbiter)。qw 交违章05 候选生成根因诊断(commit d6f2eb3),裁定"30/30 = C(prior 偏框)"。cc 两层独立复核:**底层数据可信,但因果裁定被脚本的 stage 判定顺序 bug 带偏**。

## 0. 独立复核(两层, 全部对上)
- **聚合层**(cc 从 `data/output/qw/05_candidate_gen_per_frame.csv` 自写重算, 不跑 qw 脚本): YOLO IoU≥0.3=**2/30**、HSV(_candidates)IoU≥0.3=**1/30**、prior_roi_iou>0=**0/30**、prior_contains=**0/30**、低饱和 S<130=**18/30**、`filtered_by=="sat"` 与 `S<130` **完全重合 18/18**、prior 直采颜色匹配=**15/30**。全部对上 qw 报告。
- **重建层**(cc 从零重跑 fi=0/236/1534/1947 四帧, 自算 gov 内 HSV 均值 + YOLO/HSV/prior_roi IoU, 非 import qw 的 diag_frame): S=154.9/121.8/115.2/125.8、yolo_iou=0/0/0.3310/0.9857、hsv_iou=0.1761/0.0339/0/0.2800、prior_roi_iou 全 0.0000 —— **逐帧对 CSV 到小数点后 4 位, 0 不一致**。gov_color_gt=green/green/red/red 亦对上。
- **结论: 底层数据可信**(gov 灯位置/HSV/各路命中度都实测无误)。问题只在归因裁定。

## 1. ⚠️ 揪出问题: "30/30 = C" 是 stage 判定顺序 bug 造成的误判
- 看 `scripts/diag_05_candidate_gen.py:173-188` 的归因决策树: 先查 `elif not prior_contains: stage="C"`(L179), **再**才查 `elif ms < SAT_MIN: stage="B"`(L183)。
- 而 30/30 帧 `prior_contains=0`(prior 偏框) → **每帧都在 L179 短路进 C, L183 的低饱和检查根本没被执行到**。这把"18/30 帧 gov 灯低饱和(S<130)也会独立地被 `_candidates` 的 sat_min=130 过滤"这个**共存主因彻底掩盖**。所以"30/30 C"是判定顺序的伪结论, 不是真因分布。
- **更关键的因果错位**: 天花板 10% 是在 **不用 prior 的候选口径**(YOLO cls9 + 裸 `_candidates`)上量的 → **prior 偏不偏框对"天花板为何只有 10%"在因果上完全无关**。prior 只影响生产 `observe()` 的 prior 直采路径, 不进候选池。qw 用"prior 偏框"解释一个 prior 根本没参与的指标, 是答错了半个问题。

## 2. 真实归因(cc 从同一份可信数据重新分解)
天花板 10% 的直接成因(候选池 = YOLO + 裸 HSV, 无 prior):
- **B 低饱和过滤(主因, 18/30)**: gov 灯 gov-box 内 S 均值 103–155, 其中 18 帧 <130 → 被 `_candidates` 的 `sat_min=130` 截掉, 这 18 帧无论 prior 如何都产不出好的裸-HSV 候选。
- **partial-box(12/30)**: gov 灯过了 `_candidates` 但产出的候选框 IoU<0.3(11/12 帧 hsv_iou<0.3, 最高仅 0.32)—— 连通域偏/裂, 框不齐。
- YOLO 仅 2/30 命中≥0.3(灯太小/远)。

## 3. 两个独立的真 bug(都值得修, 别混为一谈)
- **(i) prior 偏框 = 生产假绿源(高优先, 关联项目核心目标)**: prior `[0.7, 0.15]` vs gov 实际 ~`[0.58, 0.34]`(cc 核过: cx 0.252~0.667 均值 0.580, cy 0.295~0.402 均值 0.336, Y 偏 0.19 最狠), prior_roi_iou=0 ∀。**生产 prior 直采在错误位置稳定返回 green: 15/15 个 gov 真值=red 的帧, prior 直采都返回 green** → 这是 05 在红灯相位下的**主动假绿发生器**, 正是全项目要杀的 false-green bug。qw 报告点到"假绿"但低估了其意义——这不是"候选没捞到", 是"生产会误报绿"。
- **(ii) gov 灯低饱和(天花板主因, 见 §2)**: 全局降 `sat_min` 有溢出风险(会给其他 10 视频加干扰候选)。安全杠杆是 prior 直采路径(S≥60, 局部化)—— 但它要在**正确位置**才有效, 依赖 (i) 修好。

## 4. 裁定 — qw 建议方向对, 因果叙事需更正
- **05 归 (i)+(ii) 复合, 不是纯 C**: 修法 = **重定位 05 的 prior 到 ~[0.58, 0.34]**(承 [[prior-misframe-rootcause]], 第③步 prior 重定位; 手持机位 cx 0.25~0.67 跨度大, 需评估固定 prior 是否够用, 或走 `derive_priors.py` 逐帧/跟踪式)。这一修**同时**杀掉 (i) 的假绿 + 让 prior 直采(S≥60)在正确位置捞回 (ii) 的低饱和灯。qw 的位置估计 ~[0.58,0.34] cc 核过可用, 这条采信。
- **⚠️ 口径警告(qw 二元判 D 时漏掉的更深一层)**: sel_prec/天花板指标**从不跑 prior 直采路径** → **即便把 prior 修对, 天花板指标也不会动**(它只量 YOLO+裸HSV)。要让修复在指标上可见, 必须把 prior 直采候选纳入天花板/eval 口径, 否则 05 会永远读作 ~10%、误导后续判断。qw 正确排除了"当前错位 prior 直采能救"(那确实不能), 但没点出"指标本身不测这条路径"。
- **gate 关联**: 05 的 ~12 漏绿是排序救不了的硬地板(见 `2026-08-03-cc-verify-qw-recall-ceiling.md`)。修好 05 候选生成(prior 重定位 + prior 直采入池)可望捞回这 ~12 漏绿 → 直接利好 worst-seed 漏绿≤80 gate。所以这条不是孤立优化, 和 A 消融的 gate 目标同向。
- 红线不变: 只读诊断已交, 修复(改 `configs/light_priors.json` / 跑 `derive_priors.py`)是第③步独立任务(assignable to wb/后续), 非 qw; 权重不进库; cc 复核。

---
**一句话**: qw 05 诊断底层数据 cc 两层 bit-for-bit 通过(聚合 + 重建 4 帧 0 不一致)。**但"30/30=C"是脚本判定顺序 bug 的伪结论**(L179 先查 prior_contains, 30/30 短路进 C, 从没跑到 L183 的低饱和检查)。真因分解: 天花板 10% 由**低饱和过滤(18/30, S<130 被 sat_min=130 截)+ partial-box(12/30)** 驱动, prior 偏框对天花板因果无关(天花板口径不用 prior)。另揪出独立高优先 bug: **prior 偏框使生产 prior 直采在 15/15 红灯帧返回假绿 = 主动 false-green 源**。修法采信 qw: 重定位 05 prior 到 ~[0.58,0.34](第③步), 一石二鸟(杀假绿 + 捞回低饱和灯), 且利好漏绿 gate。口径警告: 天花板指标不跑 prior 直采路径, 修好 prior 也不会反映在该指标上, 需把 prior 直采纳入口径才能量到。
