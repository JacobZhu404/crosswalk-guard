# wb 归因报告: 违章01 FP 生产根因 (plan-gate #3 前置)

> 任务: cc→wb brief `docs/handoff/2026-08-05-cc-brief-wb-01fp-diagnosis.md` (87dd9ff)。
> 范围: 只读诊断, 不写生产码, 过 gate 前不建 worktree。
> 唯一二分: 01 FP 走 `violation_engine :87`(on_crosswalk+green/flashing→confirmed 误绿)
> 还是 `:90`(unknown+occluded→review)?

## 0. 结论(一句话)

**01 FP = `:87`(confirmed / green), 不是 `:90`(review)。** 它是 `observe()` 在全程红灯视频里**误读绿**(false-green), 而非"遮挡 unknown 自动进 review"。修法方向 = **修 observe 判色/排干扰**(brief 的 :87 修法), 与 :90 的"修 review 门控"无关。

## 1. 二分判定(来自生产 DAG 重跑)

走生产 DAG(`redlight.app.cli.run` → `dag.py` → `observe()` → `BatchViolationEngine.accumulate/decide` → `decide_violations`), **禁用 `select_gtfree`**。逐帧打点视频01 `[45.4, 64.3]`(窗口含 [48.4,61.3]±3s)。

重跑产出事件(与 GT 真负例矛盾 = FP):

| 字段 | 本诊断(pytorch 当前 HEAD) | cc_baseline 快照(cc2414b) |
|---|---|---|
| status | **confirmed** | confirmed |
| light_state | **green** | green |
| start_ts / end_ts | 48.36 / **61.29** | 48.36 / 60.62 |
| track_id | 129 | 121 |
| vehicle_class | car | bus |
| confidence | 0.545 | 0.506 |

- 分支映射批处理引擎 `decide_violations`: `confirmed` ← `go`(green/flashing 灯段)∩静止∩压线; `review` ← `review_light`(unknown+occluded/inferred 段)。本事件 status=**confirmed**、重叠灯段 `state=green, evidence=visible`(非 unknown/occluded) → 明确走 **`:87` 绿色分支**。
- `decide_violations` 输出中 **review 事件数 = 0** → 不是 `:90`。
- 重跑 `确认违规=1 待复核=0`, 与生产 FP 签名(confirmed/green, ~48.4–61.3)一致, 确定性可复现(cc 可 bit-for-bit 重跑本脚本)。
- track_id/vehicle_class/end_ts 与 cc_baseline 略有漂移(121/bus/60.62 → 129/car/61.29): 因 cc_baseline 快照来自更早的 cc2414b 代码, 当前 HEAD 跟踪器/车辆类标不同; **二分结论与窗口不受影响**。(建议: 若要把 cc_baseline 当回归基准, 待 plan-gate 后由 cc 刷新。)

## 2. 误绿归因(干扰源 = prior 直采处的环境绿)

### 2.1 采样路径
窗口 140 推理帧中采样路径分布: **`prior_direct_sample` = 132 (94%)**, `prior_hold_old_color` = 8。即 `observe()` 在 01 走 **prior 直采模式**(视频有先验 `light_priors.json: 违章01=[0.35,0.35,160]`), 固定采样归一 (0.35,0.35) 处 160px ROI。

### 2.2 绿从哪来: 环境绿, 非信号灯
prior ROI 内绿/红像素统计(与生产 `_sample_roi` 同阈值, 只读复算):

| 时段 | prior ROI 绿占比 g_frac | 红占比 r_frac | 绿斑质心(ROI内) | 判读 |
|---|---|---|---|---|
| 46.2–49.3s | 0.13% | 0.00% | — | 极弱绿, 因红≈0 即翻绿(阈值敏感) |
| **52.67–53.61s** | **51.0%** | 0.6% | (0.30, 0.46) | **ROI 一半是绿, 弥散** |
| **55.90–56.84s** | **46.5%** | 3.5% | (0.38, 0.60) | **ROI 近半是绿, 弥散** |

- 强绿(prior ROI g_frac≥10%)共 16 帧, 集中在上述两段; 绿斑质心弥散、占比近半 → **是弥散大块绿面(车漆/树叶/车窗反光), 绝不是紧凑灯泡**(真行人灯灯泡占 ROI <5% 且质心居中)。
- 全局亮斑大绿(g_area>5000)在窗口 80/140 帧出现 → 场景环境绿普遍; 但**决定性误绿来自 prior 直采**(132 帧走该路径), 不是全局候选。

### 2.3 根因(代码层)
`_sample_roi`(`traffic_light.py:626`)用于 prior 直采, **硬编码 `sat_min=60`**; 而候选生成 `_candidates` 用 config `sat_min=130`。即 prior 直采比主检测器**宽松一倍**。配合 `_sample_roi` 的 `min_frac=0.002`(0.2%)与 `g_frac > r_frac*1.3` 判定: 当 ROI 内红≈0 时, **只要 ~0.13% 的中饱和绿**(树叶/绿车/反光, S~60–100)就翻绿。01 全程红+部分遮挡, 真实灯或被挡或为红/灭, prior 固定位置却持续采到路过/驻留的**环境绿** → 误绿段 [48.36–61.29] → 静止车压线 → confirmed FP。

> 一句话: prior 直采的宽松饱和度阈值(60 vs 130) + 极低绿占比门槛, 把遮挡处的环境绿当成了行人绿灯。

## 3. 修法方向候选(仅方向, 不实现 — 走 plan-gate #3 再动码)

brief 的 :87 修法 = "修 observe 判色/排干扰"。具体杠杆(按预期收益排序):

1. **统一 prior 直采饱和度阈值**: `_sample_roi` 的 `sat_min` 60 → 与候选生成一致(≈130)。真 LED 绿灯 S≥200, 树叶/绿漆 S~60–100 会被拒。这是**直达根因**的一行级修正, 预期直接消除 01 误绿且不影响真绿灯读取。
2. **弥散绿拒绝**: prior 直采加空间集中度检查(绿斑 lamp_score / 质心紧凑度), 拒绝"占 ROI 近半的弥散绿", 只认紧凑灯泡状绿。
3. **遮挡期降级 unknown**: 当斑马线掩膜判定遮挡(`_is_occluded` 底边截断)且 prior ROI 无紧凑灯形时, 直采结果判 unknown 而非强制绿/红, 交下游 review(D1)而非 confirmed。
4. (次)提高 `min_frac` / 加大绿红比容差, 降低噪声绿触发率。

> 注: 上述均针对 `observe()` 判色, 与 :90 的 review 门控无关 —— 再次确认本 FP 不是 review 旁路产物。

## 4. 复现(cc 可 bit-for-bit)

- 脚本: `scripts/diag_01fp_repro.py`(只读, 复用 `cli.run` + 两个 monkeypatch 抓取逐帧状态, 无循环复制, 禁用 `select_gtfree`)。
- 跑法:
  ```
  cd /Users/jacob/personal/crosswalk-guard
  PYTHONPATH=src ./.venv/bin/python scripts/diag_01fp_repro.py --evidence
  ```
- 产物: `data/output/diag_01fp/diag_01fp.json`(逐帧归因+灯段)、`violations.csv`(重跑交叉核对)、`evidence/green_ts*.jpg`(窗口内绿帧叠框: 黄框=prior ROI 160px, 红框=YOLO 灯框)。
- 裁判 GT: `datasets/gt/events.csv`「违章01,0,66.5,red,occluded,0」→ 窗口内任何 green 读数即误绿。GT 仅裁判, 不进推理。

## 5. 红线遵守

- 只读诊断, 未改任何 `src/` 生产码; 仅新增 `scripts/diag_01fp_repro.py`(诊断脚本)。
- 禁用 `select_gtfree`; 生产灯态只认 `observe()`。
- 未建 worktree / 分支; 待 plan-gate #3 裁定修法方向后才动码。
- GT 不进推理; scoped git(禁 `git add -A`); 未碰 `ped_signal.pt` / 权重未入库。

---
*署名: wb(诊断/执行)。承 cc brief 87dd9ff + plan-gate #2(09/06 selector⑤ 幻影 HALT)。*
