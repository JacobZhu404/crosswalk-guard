# cc plan-gate #3 裁定:wb 违章01 FP 生产根因诊断(8f9c152)— 诊断 PASS(bit-for-bit),但**头号修法#1 证伪**,方向纠偏到修法#2

> 审核对象:`docs/reports/2026-08-05-wb-01fp-attribution.md` + `scripts/diag_01fp_repro.py`(commit `8f9c152`)
> 署名:cc(plan-gate #3/独立复核) — 遵 [[measurements-disagree-find-the-bug]]、[[eval-methodology-gap-overfit]]
> 结论:**wb 二分结论(:87 误绿,非 :90 review)+ 根因机制(observe prior 直采采到环境绿)cc 亲跑 bit-for-bit 复现,PASS,记 wb 一功。但 cc 独立重测证伪 wb 的头号修法#1(sat_min 60→130):01 环境绿 S 中位 ~152 ≫130,升到 130 仅把 g_frac 0.51→0.42 仍判绿 → 01 仍误绿。真杠杆是修法#2(弥散绿/空间集中度拒绝),与 05 prior 重定位反转同因(ROI 太大+面积投票无形状约束,非阈值)。**

## 1. 诊断复现(cc 亲跑 `diag_01fp_repro.py`,PASS)

| 项 | wb 报 | cc 独立复现 | 判定 |
|---|---|---|---|
| 二分分支 | :87 confirmed/green | `branch_verdict=P87_confirmed_green`,production_events status=confirmed light_state=green | ✓ |
| review 事件数 | 0 | `确认违规=1 待复核=0` | ✓ |
| 误绿段 | [48.36,61.29] | light_segments green [48.36→61.29],前后皆 red(0-48.36/61.29-66.41)| ✓ |
| 采样路径 | prior 直采 94% | 窗口行全 `path=prior_direct_sample`,prior=[0.35,0.35] roi 160px | ✓ |
| 强绿帧 | 16 | `window_prior_roi_strong_green_frames=16` | ✓ |
| 翻绿机制 | 红≈0 即翻绿 | ts45.53 红(g3/r180)→ ts46.20 绿(g34/**r0** g_frac 仅 0.0013)| ✓ |

**二分结论坐实:01 FP = observe() 在全程红视频里 prior 直采误读绿(:87),非遮挡 review 旁路(:90=0)。** 与 GT 真负例(全程红 0-66.5)矛盾 = 真生产 bug(非 selector⑤ 那种非生产幻影)。**记 wb 一功。**

## 2. ⚠️ 头号修法#1 证伪(cc 独立重测,measurements-disagree)

wb §2.3/§3.1 断言:根因=`_sample_roi` sat_min=60 vs 候选生成 130 不对称,修法#1"sat_min 60→130,树叶/绿漆 S~60-100 会被拒"是"直达根因的一行级修正"。

**cc 抽两强绿帧,用生产同口径 ROI[368,171,528,331] 重算 g_frac(sat_min=60 结果 0.5100 与生产/报告 51.0% 精确吻合,口径可信):**

| 帧 | g_frac@60 | g_frac@130 | g_frac@200 | 绿相素 S 中位 |
|---|---|---|---|---|
| 52.67s | 0.510 | **0.422** | 0.002 | **155** |
| 55.90s | 0.465 | **0.402** | 0.003 | **152** |

- **环境绿 S 中位 ~152,不是 wb 说的 60-100** → sat_min 60→130 只把 g_frac 降到 ~0.42,仍 ≫ r_frac×1.3(r≈0.006)→ **仍判绿 → 01 仍误绿**。**修法#1 无效,证伪。**
- 需 sat_min≈200 才杀到 0.002,但那是大跳变,可能误杀真 LED 绿(01 全程红无真绿帧,安全性无法在 01 上自证),风险高。

## 3. 真根因 + 修法方向(cc 纠偏)

真因**不是阈值不对称**,而是(与 [[selection-precision-ranking-bottleneck]] 记的 05 prior 重定位三臂反转**同因**):
1. **160px ROI 太大**——灯泡仅占 ROI <5%,却把车漆/树叶/反光的大块弥散绿(占 ROI 42-51%)一并纳入;
2. **`g_frac>r_frac×1.3` 面积投票无形状/集中度约束**——大块弥散绿在像素占比上碾压,无论饱和度高低。

∴ 采信 wb **修法#2(弥散绿/空间集中度拒绝)为主杠杆**:真灯泡紧凑(质心居中、占比小),环境绿弥散(质心偏、占比近半)——用 lamp_score/连通域紧凑度判别,**与饱和度无关**,对 01 这种高饱和环境绿也有效。修法#3(遮挡期降级 unknown)可作二线兜底。**修法#1(sat_min→130)降级/剔除**(证伪);若坚持阈值路线须用 ~200 且必须在真绿视频回归证不杀真绿。

## 4. 裁定 + 下一步

- **诊断 PASS**(二分 + 机制 bit-for-bit 复现)。**wb 转入修法方案设计(plan-gate #4 方案 gate 前置)。**
- **方案硬条件**(cc 预告,写进方案 gate):
  1. 修法必须在 **01 上实证消除误绿段**(48.36-61.29 不再判绿);
  2. 必须在**真绿视频(05/06/09 等)回归证不杀真绿**——01 无真绿帧,安全性只能靠真绿视频兜底(否则重蹈"杀真绿"覆辙,见 [[light-classifier-retrain]] ped_signal.pt 屠真绿教训);
  3. 端到端 **F1 不回退**(01 FP 消除应抬过 0.889,且 06/09 TP 不掉);
  4. 优先形状/集中度判别(修法#2),阈值路线(修法#1/sat_min)已证 130 无效、200 高风险,不作首选。
- 红线沿旧:先方案 gate 再动码 / 隔离 worktree / scoped git / GT 只诊断不进推理 / 不碰 ped_signal.pt·权重不入库 / 原子写。

## 5. 一句话给 Jacob
wb 的 01 FP 诊断(全程红却在 48.4-61.3 误绿开罚单=生产 observe prior 直采采到环境绿,:87 非 :90)cc 亲跑全复现,**是真生产 bug、记一功**。但 cc 独立重测**证伪了 wb 的头号修法**(sat_min 60→130):01 环境绿饱和度中位 ~152 远超 130,升阈值仍判绿。真杠杆是**弥散绿的空间形状判别**(与 05 prior 反转同因:ROI 太大+面积投票无形状约束)。诊断放行,修法方向已纠偏,下一步 wb 出修法方案走 plan-gate #4,方案必须在真绿视频回归证不杀真绿。

---
*署名:cc(plan-gate #3/独立复核)。诊断证据=亲跑 diag_01fp_repro.py bit-for-bit;修法#1 证伪证据=生产同口径 ROI 重算 g_frac@60/130/200 + 绿相素 S 中位 152。承 [[selection-precision-ranking-bottleneck]] 05 反转、[[prior-misframe-rootcause]]、[[light-classifier-retrain]] 屠真绿教训。*
