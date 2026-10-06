# 违章01 FP 修法方案 (plan-gate #4 预审)

> 上游: plan-gate #3 (commit 059f650, PASS) — cc 亲跑 `diag_01fp_repro.py` 复核 :87 confirmed/green 误绿为**真生产 bug** (记 wb 一功); 修法#1 (`sat_min→130`) 被 **证伪** (01 环境绿 S 中位 ~152 ≫ 130, 升到 130 仅 g_frac 0.51→0.42 仍判绿)。
> 本方案为 cc plan-gate #4 评审稿。过 gate 前 **不写生产码、不建 worktree**。

## 1. 根因(再定位, 与 05 prior 反转同课)
`TrafficLightDetector._sample_roi` (`traffic_light.py:626`) 对固定 160px prior ROI 做 HSV 绿/红像素计数, 以 `g_frac > r_frac*1.3` **面积投票**判绿, 无任何形状/空间约束。01 的 prior=`[0.35,0.35,160]` 在遮挡/红灯期 ROI 内只有扩散环境绿(树叶/车身反射, S 中位 ~152), 无紧凑灯斑 → 仍被判绿 → 误绿段 [48.36–61.29] → 静止车占道 → **confirmed FP**。

关键认识: **问题不在饱和度, 而在"面积投票无形状约束"**。160px ROI 太大 + 只要绿像素够多就判绿 = 扩散环境绿被当灯绿。饱和度阈值(修法#1)无效, 因为环境绿本就够饱和。

## 2. 修法设计 — 修法#2: 紧凑绿斑门控
### 2.1 改动面(最小、定向)
- **只改** `_sample_roi` 的绿判定 (唯一 bug 路径)。
- **不改** `_sample_box` (YOLO 框本身是形状过滤, 框内绿即灯绿, 无面积投票 bug)。
- **不改** 候选 `_candidates` (基于局部亮斑, 隐式紧凑)。
- 保留 160px ROI 大小 (形状门控已解耦"ROI 太大"问题; 缩 ROI 作可选二级, 不纳入本次 scope, 需独立回归)。

### 2.2 新增 helper
`_green_has_compact_blob(g_mask, total, tau, min_lamp_px) -> (bool, solidity, area)`:
1. `cv2.connectedComponents`(或 `findContours`) 取 `g_mask` 连通域。
2. 对每个连通域算面积 `area` 与 bbox 紧实度 `solidity = area / (bbox_w * bbox_h)`。
3. 若**任一**连通域满足 `solidity >= tau` 且 `area >= min_lamp_px` → 返回 `(True, best_solidity, best_area)` (存在灯状紧凑绿斑); 否则 `(False, max_solidity, max_area)`。
4. 返回最紧凑块指标供诊断记录。

### 2.3 `_sample_roi` 绿判定改写(草图)
```python
# 原: if g_frac > r_frac * 1.3: return "green", g_n, r_n
# 改: 仅当面积占优 且 存在紧凑绿斑, 才判绿
compact = self._green_has_compact_blob(g_mask, total, tau, min_lamp_px)
if g_frac > r_frac * 1.3 and compact[0]:
    return "green", g_n, r_n
# 绿被拒(扩散环境绿)→ 当"无绿"处理, 走原 red/无 分支
if r_frac > g_frac * 1.3:
    return "red", g_n, r_n
# tie-break: 绿不紧凑 → 不判绿; 有红判红, 无红回退 unknown/off
if r_frac >= g_frac:
    return "red", g_n, r_n
return None, g_n, r_n   # → 调用方(_sample_prior_color)返 None → observe 回退 near 候选 / off→unknown
```
对 01: 遮挡期无紧凑绿斑 + 红≈0 → 返 `None` → `off` → `unknown` → 不 confirmed → **误绿段消除**。

### 2.4 为何不杀真绿 (条件②)
- 真绿灯 = 紧凑亮圆斑 → `solidity` 高 + 面积达标 → 通过。
- 环境绿 (树叶/反射) = 大面积扩散、无紧凑块 → `solidity` 低 → 拒绿。
- 用 **"任一连通域满足"** 而非 "仅最大块": 即使 ROI 混有树叶, 只要存在灯状紧凑块仍判绿 → **避免 ped_signal.pt 式屠真绿** (环境绿 + 真灯共存时保真绿)。

## 3. 阈值推导协议(写码前先量, 不盲定)
扩展 `diag_01fp_repro.py` (只读) 增加 per-frame 紧凑度度量:
- 01 强绿帧 (52.67–53.61 / 55.90–56.84): 算最大绿块 `solidity`/`area` → 期望 **low** (扩散)。
- 05/06/09 真绿帧: 算同指标 → 期望 **high** (紧凑)。
- 选 `tau` 使 01 全拒、05/06/09 全过, 留 margin。初值 `tau=0.35`, `min_lamp_px=25` (按 160px ROI 试, 以实测校准)。
- 同步记录 red 块紧凑度, 确认 red 判定无需门控 (01 红≈0, 不动 red 分支以避回归; 仅监控)。

## 4. 硬条件验证 (plan-gate #4 放行门槛)
| 条件 | 验证方式 | 通过判据 |
|---|---|---|
| ① 01 消误绿 | 重跑 `diag_01fp_repro.py` + 01 端到端 | 误绿段消失; 生产 `违章01` 违规=0 / 待复核=0 (或转 unknown 不 confirmed) |
| ② 05/06/09 不杀真绿 | 三视频端到端重跑 | 真绿帧保留 (green 计数 / 误绿率不恶化, 绿帧数 ≥ baseline) |
| ③ 端到端 F1 不回退 | 全 11 视频端到端 | confirmed/漏绿/误绿 F1 ≥ 当前 HEAD 基线 (v2+box) |
| 附带 | 02/03/04/07/08/10/11 回归 | 无新增 FP / 漏绿 |

## 5. 红线(全程)
- 过 gate 前不建 worktree/分支、不写生产码 (本方案文档除外); 阈值推导用只读诊断脚本。
- 禁 `select_gtfree`; 不碰 `light_priors.json` 坐标; 不加 per-video 灯参; 不碰 `ped_signal.pt`; 权重不入库。
- scoped git (禁 `git add -A`); 诊断产物 gitignored。

## 6. 交付物
- 本方案: `docs/plans/2026-08-05-wb-plan-gate4-01fp-fix.md`
- 过 gate 后(待 cc scope 确认):
  1. 改 `traffic_light.py::_sample_roi` + 新增 `_green_has_compact_blob` helper
  2. 扩展 `diag_01fp_repro.py` 记录紧凑度度量
  3. scoped commit (仅上述文件)
  4. 三硬条件证据表 (①误绿消失 ②05/06/09 真绿保留 ③全量 F1 不回退) 交 cc 复核
