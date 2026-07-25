# 灯态标注集诊断与修复指引（light_feedback.csv, 376 帧）

> 用途定位：**误差诊断 + 调参指引**，**不做训练集**。
> 数据来源：`data/output/annotated/light_feedback.csv`（灯态误差画廊人工标注，全部为 pred≠GT 的 mismatch 帧）。
> 生成脚本参考：`scripts/analyze_light_feedback.py` / `analyze_light_feedback2.py`

---

## 1. 这份标注能不能当评测集？

**能当「误差诊断 / 回归校验集」，不能当「准确率基准集」。**

- ✅ **误差诊断**：376 帧全是人类确认过的 pred≠GT 错例，且带 `verdict`(谁错) + `reason`(哪类错) + `note`(具体现象)。这是修算法的金矿。
- ✅ **回归校验**：把这 376 个 `(video, t_sec, frame_idx)` 跑当前管线，统计「修复后还有多少仍错 / 错在哪类」，可量化每次改动收益。
- ❌ **准确率基准**：画廊只采样 mismatch 帧（pred≠GT 才进画廊），所以**上任何算法准确率都被构造为 0%**，无法衡量"整体变好"。要算真准确率，需另抽一份**随机/黄金帧**（含 pred=GT 的样本）。
- ⚠️ **不是训练集**（详见 §5）。

**可用性结论**：直接用于"指导修哪、修完验证"，不要用于"喂模型训"。

---

## 2. 标注总体构成

| 维度 | 分布 |
|------|------|
| 总帧数 | 376 |
| verdict | algo_wrong 334 (88.8%) / other 41 (10.9%) / both_wrong 1 (0.3%) / **label_wrong 0** |
| reason | other 271 / reading_point 67 / search_area 38 / color 0 / gt_flipped 0 |
| 视频 | 违章03 281 (75%) / 违章02 89 / 违章04 6 |

**关键信号**：
- `label_wrong = 0` → 这批的 GT 本身**可信**，没有"算法其实对、GT 标反"的情况。可放心用 GT 当裁判。
- `color = 0` 且 `gt_flipped = 0` → **没有"位置对但颜色读翻"或"GT 段界标反"的 case**。说明问题不在颜色分类、不在 GT 边界，而在**定位 / 置信度**。
- `reason=other` 占 271，但备注揭示了它其实高度集中（见 §3）。

---

## 3. 误差聚类（读备注后的真实分类）

### 聚类 A — 过度自信：该出 `unknown` 却出了确定态  ⭐最高优先
- **276 帧**（占全部 73%）备注明确"应输出 unknown"。
- 细分：
  - 违章03 遮挡段 264 帧：红灯相位、灯被公交车/立柱遮挡 → 人类"可以输出 unknown"
  - 违章02 108–110s 7 帧：画面里**根本没有红绿灯** → 人类"输出 unknown 也可以"
  - 违章04 5 帧：类似遮挡
- 这些帧的算法预测：red 154 / green 111 / flashing 11（全是"自信地错"）。
- **印证 E12 安全语义**：unknown = 不commit违章（安全侧）。遮挡/无灯时出 unknown 正是项目要的设计。

### 聚类 B — 读取点（黄圈）偏离真灯  ⭐高优先
- 明确 `reason=reading_point` 67 帧 + red→green 备注里"黄圈右上方/右边/左方"约 24 帧 ≈ **91 帧**。
- 备注范式："真实的交通灯在黄圈的**左方/上方/右上方**"。
- 黄圈目前取的是**搜索框中心或固定偏移**，不是真灯灯头质心 → 取到的是背景/相邻灯 → 颜色自然读错。

### 聚类 C — 搜索区（蓝框）太窄，真灯在框外  ⭐中优先
- `reason=search_area` 38 帧，**全部在违章02**。
- 备注范式："真实的交通灯在**蓝框的左边**"。
- 蓝框由 prior/车辆位置推导，margin 太小 → 真灯在框外未被搜索。

### 聚类 D — red↔green 系统性反向（需谨慎解读）
- 混淆矩阵：pred=red→gt=green **212** + pred=green→gt=red **88** = 300 帧。
- **但**：其中绝大部分可被 A/B/C 解释（遮挡该 unknown、读取点偏、蓝框窄）。且 **0 个 `color` 原因** → 这不是"位置对但色翻"，而是**定位错导致的采样错**。
- 结论：**先修定位（B/C）和置信度门（A），再回头看 red↔green 是否消失**；不要一上来动颜色分类器。

---

## 4. 修复优先级与代码落点

### P0 — 加「遮挡/无信号 → unknown」置信度门  （覆盖 276 帧 / 73%）
- 落点：`traffic_light.py` 的 `observe()` 单帧入口 + `TemporalFusion` 的 unknown 传播。
- 做法：
  1. 搜索区内若**无清晰灯头 blob**（亮度/饱和度峰值低于阈值）→ 直接 `unknown`；
  2. 检测到遮挡线索（灯头被部分遮挡、置信度低）→ `unknown` 而非强行给状态；
  3. TemporalFusion 已有 unknown 分段逻辑，确保**不被 prior 强填成确定态**（这正是违章03 prior 有害的根）。
- 验证：跑这 276 帧，修复后 majority 应为 unknown（符合人类"可以 unknown"）。

### P1 — 读取点改为「灯头 blob 质心」  （覆盖 ~91 帧）
- 落点：`traffic_light.py` 读取点计算。
- 做法：在搜索区内用 HSV 亮斑 + 饱和度找**最强灯头连通域**，取该域质心作为读取点（而非框中心/固定偏移）。可借 M1 计划的"YOLO灯框 ∪ HSV亮斑候选"候选并集。
- 验证：这 91 帧读取点应与真灯重合；重算颜色后 red↔green 应大幅下降。

### P1 — 搜索区加宽 / 候选并集  （覆盖 38 帧，全违章02）
- 落点：`traffic_light.py` 搜索框推导（prior + 车辆位置）。
- 做法：蓝框 margin 放大（如上下左右各 +30%），或改用 M1 候选检测器输出兜底——只要任一候选落在框外附近也纳入。
- 验证：这 38 帧真灯（蓝框左侧）应落入搜索区。

### P2 — 修完 A/B/C 后回归重测，再决定是否动颜色分类
- 重跑全部 376 帧：统计修复后剩余 error 的 reason 分布。
- 若 red↔green 仍显著且位置已对 → 才考虑颜色分类 refinement；当前证据**不支持**直接改颜色。

---

## 5. 为什么「不做训练集」（明确反对）

1. **样本有偏**：全部是 mismatch 帧，无 pred=GT 样本 → 训出来的模型会过拟合"难例"，在正常帧上反而退化。
2. **标签与 GT 冲突**：聚类 A 的 276 帧，人类说"应 unknown"，但 GT 标的是 green/red（相位真值）。若拿去训，模型会被教"遮挡时也该给 green/red"，与 E12 安全语义（unknown=安全侧）**直接矛盾**。
3. **M1 训练需要的是另一份数据**：CC 已重抽 `datasets/ped_signal/`（正负 1:1 平衡 + 校验画廊），那份才是 ped_classifier 的训练原料。本标注集的定位是**修 pipeline 的定位/置信度模块**，不是训分类头。
4. **结论**：本集 → 诊断 + 回归校验 + 调参指引；训练 → 用 CC 的 crop 集。

---

## 6. 建议的评测/回归用法

`scripts/eval_light_fast.py` 已有 `--regression` 接口（接收 `video,t_sec,expected_state` CSV）。我已从本标注派生好回归用例：

**派生规则**（写入 `data/output/light_eval/regression_from_feedback.csv`）：
- 备注含 "unknown" / "遮挡" / "没有红绿灯" → `expected_state=unknown`（聚类 A，修完应出 unknown）
- 其余 algo_wrong/both_wrong → `expected_state=gt`（聚类 B/C，修定位后应得正确相位态）
- `label_wrong` 帧（本集为 0）排除

```bash
# 1) 现状基线：用本集跑回归，看当前通过率 / 各类剩余错误
python scripts/eval_light_fast.py --regression data/output/light_eval/regression_from_feedback.csv

# 2) 修完 P0/P1 后重跑同一条命令，对比通过率提升
#    （脚本已能对 regression CSV 逐帧取 pred 与 expected 比对）
```

> 注：回归集是 mismatch 采样，基线通过率天然偏低（这是设计使然），**看的是"修复后剩余错误率下降"而非绝对通过率**。真准确率仍需另抽随机/黄金帧。

---

## 7. 一句话总结给调参同学

**不要动颜色分类器。** 当前 73% 的错是"遮挡/无灯时太自信没出 unknown"，其余是"读取点偏"和"搜索框太窄"——全是**定位与置信度**问题。先把 P0（unknown 门）和 P1（读取点质心化 + 搜索框加宽）做了，再回来用这 376 帧做回归校验，red↔green 大概率大幅收敛。
