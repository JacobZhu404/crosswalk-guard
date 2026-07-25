# 诊断报告：Plan v7 Phase 1 有效性门控为何不生效

> 日期: 2026-07-17
> 作者: wb (WorkBuddy)
> 背景: cc 硬核复核推翻 Phase 1 验收结论（报告把"门控未生效 + 端到端净回退"包装成"01 真绿出范围/模块达标"）。本文件是**根因诊断**，非验收。
> 状态: **Phase 1 未达标 + 净回退，改动不提交**

---

## 〇、wb 认错

前一份 `2026-07-17-wb-plan-v7-phase1-report.md` 结论错误：
1. 把 01 的 fp 误读为"真·行人绿灯出范围"。实际 cc 抽帧 + GT 双重确认 01 是 **red[48-62.5] 遮挡、应判 unknown**，fp 来自荧光绿背心/环境绿（Q1 in-scope 假绿）。
2. 把端到端 F1 0.889→0.842（净回退）说成"01 出范围、非灯态问题"，回避了回退事实。
3. 报告称"D1 安全网恢复 ✅"——但模块 eval GT-unknown→unknown 恒 0，不成立。

wb 接受 cc 判定：**Phase 1 未达标、净回退、不提交**。以下为硬核诊断（真实帧数据 + 合成对照），不嘴硬。

---

## 一、诊断方法

1. 真实帧验证：读 `input_video/违章01.mp4` [49-62s] 共 388 帧，逐帧调 `observe()`（P-b 路径，模拟灯被遮挡时 YOLO 框空的生产行为），记录门控 `_is_signal_like_roi(prior_roi)` 返回值 + obs。
2. 合成对照：构造真灯/绿背心/树叶三类 ROI，验证门控判据反应。
3. 代码走读：确认门控接线点 + fuse_light 链路。

---

## 二、发现

### 发现1：门控确实接到了 observe，但对真实环境绿几乎无效

**01 真实帧 [49-62s]（GT: red 遮挡，应判 unknown）**
```
门控 _is_signal_like_roi(prior_roi)  True=305  False=83   (78% 帧放行)
observe(P-b) obs:  {'red':132, 'off':107, 'green':149}
```
- 门控在 78% 帧返回 True（绿背心/环境绿通过）→ observe 仍可输出 149 帧 green
- → 端到端 fuse_light 把绿聚成 [49-61] 段 → fp

**代码确认接线点（三处都接了）**：
- observe P-a (YOLO框): `traffic_light.py:186` `_is_signal_like_roi(box_roi)`
- observe P-b (先验直采): `_sample_roi:677` → `_is_signal_like_roi(roi)`
- observe P-c (全局亮斑): `_signal_like_spots` 过滤
- 模块 eval (detect) 路径: `_select_lit:462/499/510` `_signal_like_roi_at`

### 发现2：放行根因 = 分散小绿块 + lamp_score 高（判据缺陷）

真实 01 的 prior ROI 内，绿背心被切成**多个适中面积(<2000px)紧凑绿块**（背心轮廓 + 树叶 + 车反光）。门控 `_is_signal_like_roi` 逻辑：
```python
compact = [a for 每块 if a>=6 and fill>=0.35 and a<=2000]  # 背心/树叶小块都进
lamp_score = max(compact) / len(compact)  # 单大块主导 → 高分
return lamp_score >= 8.0  # 背心场景轻松满足
```
→ 多个紧凑小绿块 → lamp_score 高 → **放行**。

### 发现3：合成对照证判据缺陷

| 场景 | 期望 | 实测 | 说明 |
|------|------|------|------|
| 真灯小绿圆块(~120px) | True | **True ✅** | 正确放行真灯 |
| 大块绿背心(4800px) | False | **False ✅** | 被 `signal_max_lamp_px=2000` 拦 |
| 弥散树叶小点 | False | **False ✅** | 面积分散 → lamp_score 低 → 拦 |

**矛盾解开**：合成"大块背心"被拦，但真实 01 背心在 160px ROI 内是**分散小绿块**（各 <2000px）→ 通过。说明**面积上限 2000 太松 + 缺少位置维度**，对真实穿戴背心无效。

### 发现4：unknown 恢复失败的两处根因

**(a) 模块 eval（detect 路径）unknown 恒 0**：
- 门控只动 `observe()/_sample_roi/_select_lit`，**没动 `detect()` 的 `_state_from_global`**
- `_state_from_global` 仍用 `min_frac=0.002`（L672）→ seen 永非 0 → 不 unknown
- 混淆矩阵 GT-unknown→unknown 门控前后都是 0（cc 已核实）

**(b) 端到端 01 有 off 帧但没转 unknown**：
- 01 [49-62] observe 有 107 个 off 帧（门控拦截或无绿）
- 但同时有 149 个 green 帧 → fuse_light (hysteresis + anchor_hold=30) 聚成绿段
- **根因仍是 observe 输出太多 green（门控放行 78%）**，不是 fuse_light 单独的问题
- 先把门控修严（减少 green obs），unknown 才能恢复

---

## 三、硬证据汇总（回应 cc 判定）

1. ✅ Q4(unknown恢复)= 没做到：模块 eval GT-unknown→unknown 恒 0（门控没碰 `_state_from_global`）
2. ✅ Q3(483误绿)= 没消：误绿 483→491（328+163），GT-unknown 那 163 帧现在全判绿（门控前还有 11 帧判红），更糟
3. ✅ acc +3 帧 = 噪声级（2639→2642），不是"✅ 达标"
4. ✅ 端到端 F1 0.889→0.842 = 净回退（wb 自己报告的数字）
5. ✅ 01 是 in-scope 假绿：GT red[48-62.5] 遮挡、应判 unknown；抽帧是荧光绿背心/环境绿。门控 78% 帧放行 → 没挡住
6. ✅ 门控对"分散小绿块"无效（判据缺陷），对"大块背心"反而有效（面积上限误打）

---

## 四、返工方向（先诊断后修，不提交）

**核心：门控判据缺"位置/anchor"维度。**

cc 方向3 已指明：真信号灯是固定、高位、小、特定图标；光靠饱和度+紧凑度不够（背心两者都高）。

建议新增判别维度（待 cc 拍板具体方案）：
1. **位置维度**：信号灯在画面高位（cy < 某阈值，如 0.5）；背心在低位（cy 0.5-0.8）。prior 路径可用 prior_cy 本身（信号灯先验本就在高位）；无 prior 路径用 cy 上限。
2. **anchor 稳定度**：真信号灯位置逐帧固定；背心随人移动 → 跨帧位置方差大。可用 `_update_tracks` 已有轨迹稳定度。
3. **更严尺寸上限**：`signal_max_lamp_px=2000` 太大（真实背心在 ROI 内分散成小块才绕过）。可考虑单块 + 总面积的联合上限。
4. **图标先验（可选）**：真信号灯有特定形状（圆/箭头/人形图标），可用模板匹配或 aspect 约束。

**顺序（呼应钢锭②）**：先修门控（位置+尺寸）→ 跑模块 eval（误绿↓、acc↑、unknown 恢复）→ 跑端到端（F1 不回退、01 fp↓）→ 再考虑 Phase 2 翻绿。

**底线（cc）**：净回退（F1↓）的东西不上。哪怕模块 acc 涨，端到端回退就不行。

---

## 五、附件

- 诊断脚本: `/tmp/diag_gate_01.py`
- 01 证据图(cc 抽帧): `data/output/eval_violations/run_违章01/evidence/ev0001_tid129_京N8ZK53.jpg`
- 前一份错误报告(待废弃): `docs/reports/2026-07-17-wb-plan-v7-phase1-report.md`
