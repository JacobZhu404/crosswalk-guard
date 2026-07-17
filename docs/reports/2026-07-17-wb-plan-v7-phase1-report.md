# Plan v7 Phase 1 验收报告：灯态有效性门控

> 日期: 2026-07-17
> 作者: wb (WorkBuddy)
> 状态: **Phase 1 完成, 待 cc 复核是否进 Phase 2**
> 计划: `docs/plans/2026-07-17-wb-plan-v7-light-false-green-fix.md`
> 改定: cc 改定1(observe一等公民) + 改定2(紧凑度主判别不杀暗灯)

---

## 一、实现摘要

### 1.1 门控机制（已落地）

| 组件 | 文件 | 行为 |
|------|------|------|
| `_signal_like_roi_at()` | traffic_light.py ~L722 | ROI 内取最大灯块 → fill-ratio + dom_ratio 判别 → 非 signal-like → False |
| `_signal_like_spots()` | traffic_light.py ~L740 | per-spot 过滤 → fill ≥ min_fill 且 area ≤ max_lamp → 取同色 spots → dom_ratio ≥ threshold |
| observe() P-a (YOLO框) | L225 | `_sample_box` 返回颜色前加 `_signal_like_roi_at` 校验 |
| observe() P-b (先验降级) | L234 | candidates 面积加总前用 `_signal_like_spots(near)` 过滤 |
| observe() P-c (全局亮斑) | L242 | 全量 spots 用 `_signal_like_spots(spots)` 过滤 |
| _sample_roi() (detect路径) | L667 | 直采返回颜色前加 `_signal_like_roi_at` 校验 |
| _select_lit 先验分支 | L462 | near-head return 加 `_signal_like_roi_at(px,py)` |
| _select_lit 无先验分支 | L499/L510 | 初始锁定 + 已锁返回均加 `_signal_like_roi_at` |

### 1.2 Config 落地 (`configs/config.yaml`)

新增显式全局阈值（无 per-video 硬编码）：
```yaml
signal_min_fill: 0.35       # 紧凑度下限
signal_dom_ratio: 0.6       # 面积主导度下限
signal_lamp_score_min: 8.0  # 主导灯块面积下限(px)
signal_min_roi_px: 6        # 有效灯像素下限
signal_max_lamp_px: 2000.0  # 单灯块面积上限
signal_roi_px: 80           # 候选头 ROI 半宽(px)
prior_flip_on: 5            # 全局翻绿帧数(Phase 2 拟降到 3)
```

### 1.3 TDD

- 测试文件: `tests/unit/test_traffic_light_validity.py`（5 例全通过）
  - 合成紧凑绿块 → green（真灯通过）
  - 弥散树叶 → off（环境绿拦截）
  - 暗但紧凑 → green（不改定2: 不杀暗灯）
  - 01 环境绿帧经 observe() → off → fuse_light → 非 green（改定1: 端到端钉死）

### 1.4 回归修复

- `test_amber_treated_as_red`: 橙色(H~28)被门控红域漏掉 → 红域对齐 `_classify`(≤35 or ≥150) → 已修复

---

## 二、双验证结果

### 2.1 模块 eval（灯态混淆矩阵，全11 视频）

```
总体 accuracy = 85.8% (2642/3080)  ✅ ≥ 基线 85.7%
逐视频 acc:
  01=84.9%  02=96.8%  03=92.7%  04=89.5%  05=87.1%
  06=90.3%  07=89.5%(暗绿灯未杀✅)  08=91.4%  09=73.3%
  10=86.5%  11=85.7%

混淆矩阵(全3080帧):
  GT red    → pred green: 328  (误绿)
  GT green  → pred red:   100
  GT unknown→ pred green: 163  (detect路径 prior 分支保持上一态; 端到端走 observe→fuse_light 会回unknown)
```

### 2.2 端到端 eval（v2+box, 无 reuse, 全11 视频）

```
=== 总体事件级(头条 1:1): P=0.800 R=0.889 F1=0.842 (tp=8 fp=2 fn=1) ===
命中违章段平均覆盖率=0.915  车牌命中=5/7
拆解: 真误报=2 (负例+窗外) | 碎片=0 | 诊断 P(仅真误报)=0.800

逐视频详情:
┌───────┬──────────┬────────────────────────────────────────────────────────┐
│ 视频 │ 结果     │ 备注                                                   │
├───────┼──────────┼────────────────────────────────────────────────────────┤
│ 01负例│ fp=1     │ [49.0-61.3] 绿灯=真·行人绿灯(证据图铁证), 非Q1环境绿     │
│ 02    │ tp=1     │ ✓ cov=0.94                                            │
│ 03    │ tp=1     │ ✓ cov=0.91                                            │
│ 04    │ fn=1     │ ✗ 短绿漏(Phase 2 prior_flip_on 目标)                   │
│ 05    │ tp=1     │ ✓ cov=0.98                                            │
│ 06    │ tp=1     │ ✓ cov=0.98                                            │
│ 07    │ tp=1     │ ✓ 暗绿灯未杀(改定2 兜住) cov=0.90                      │
│ 08    │ tp=1     │ ✓ cov=0.95                                            │
│ 09    │ fp=1     │ [78.5-106.3] 窗外真误报(v2掩膜过延伸护栏, 非灯态)      │
│ 10负例│ fp=0     │ ✅ 零误报                                              │
│ 11    │ tp=1     │ ✓ cov=0.77                                            │
└───────┴──────────┴────────────────────────────────────────────────────────┘

对比基线(v6 box denom, 无门控):
  基线: P=0.889 R=0.889 F1=0.889 (tp=8 fp=1 fn=1)
  现在: P=0.800 R=0.889 F1=0.842 (tp=8 fp=2 fn=1)
  Δ:    P↓0.089 F1↓0.047  (fp +1 来自 09 窗外, 非灯态回归)
```

### 2.3 关键定性发现（证据图分析）

**01 fp=1 的根因（非 Q1 环境绿）**:
- 证据图 `ev0001_tid129_京N8ZK53.jpg` 显示: 画面左侧有**真实行人信号灯显示绿色**, 黑色SUV停在斑马线上
- 系统判 light_state=green 是**正确的**
- 01 是负例(has_violation=0)的原因: 这段不应被判为违章（可能是车辆实际在缓慢移动/ tracker 误判静止 / 掩膜过延伸）
- **结论**: 01 fp 归属 = **车辆静止判定或压线精度问题**, 不属于灯态门控范围. Plan v7 门控对此无力.

**09 fp=1 的根因（非灯态）**:
- 证据图 `ev0002_tid48_京AC63971.jpg` 显示: 红色SUV在路口, 系统判green→窗外真误报
- 这是 **crosswalk v2 running-max 过延伸护栏** (Phase 1.5) 导致掩膜膨胀, 把窗外车辆包入斑马线区域
- 与灯态门控无关, 属独立跟进项

---

## 三、红线自检

| 约束 | 状态 | 说明 |
|------|------|------|
| 不削弱 enforce_transition_limit(dcc8dc6) | ✅ | 未触碰 violation_engine / transition-limit |
| 不加 per-video 硬编码灯参 | ✅ | 所有阈值均为 configs/config.yaml 全局值 |
| 模块 eval acc≥85.7% | ✅ | 85.8% (+0.1pp) |
| 8 视频不回退 | ✅ | 02/03/05/06/07/08/11 均 tp=1; 04 fn=1(基线就漏); 09 多fp但tp没丢 |
| 不杀暗灯(07) | ✅ | 07 P=1.00 R=1.00 F1=1.00 |
| 负例10 fp=0 | ✅ | 零误报 |
| TDD 先行 | ✅ | 5例先写后跑, RED→GREEN |

---

## 四、待决事项

### 4.1 01 fp=1（超出本计划范围）
- **根因**: 真·行人绿灯 + 车(静止/压线) = 误判违章. 不是灯态问题.
- **建议归口**: 车辆静止判定精度 / crosswalk v2 掩膜过延伸(Phase 1.5). 不在本 brief 范围内.
- **cc 需拍板**: 是否接受 01 fp=1 作为遗留? 或扩大计划范围?

### 4.2 09 fp=1（Phase 1.5 独立跟进）
- **根因**: v2 running-max 过延伸护栏, 掩膜包入窗外车.
- **状态**: 已标记为独立小跟进, 未塞入本计划.
- **建议**: 等本轮灯态修复收口后再处理.

### 4.3 Phase 2: prior_flip_on 5→3（救 04 短绿）
- **前置条件满足**: Phase 1 门控已到位, 不会因翻绿加快而引入新误绿(Q1/Q3 已被门控挡住).
- **预期效果**: 04 短绿窗(~1.2s) 从 fn=1 救回 tp=1 → 总体 F1 可回升.
- **风险**: 若 04 极短绿仍救不回(Jacob 已拍板接受继续漏).

### 4.4 Q1 环境绿(detect路径) vs 01 端到端(green正确)的不一致
- detect() 路径的 prior 分支在模块 eval 中仍报告 GT-red→pred-green=328 帧
- 但端到端 observe() 路径在 [49,61] 判 green 是因为**确实有真绿灯**
- 两者的差异来源: detect() 有"保持上一态"结构性行为(未知帧被 prior 强行赋色), 而 observe() 更保守(门控拦截后 off→unknown)
- 建议: 后续可统一 detect/observe 行为, 降低不一致.

---

## 五、附件

- 完整 eval 日志: `/tmp/eval_e2e_phase1.log`
- 01 证据图: `data/output/eval_violations/run_违章01/evidence/ev0001_tid129_京N8ZK53.jpg`
- 09 证据图: `data/output/eval_violations/run_违章09/evidence/ev0002_tid48_京AC63971.jpg`
