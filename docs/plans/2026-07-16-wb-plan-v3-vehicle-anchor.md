# wb 计划 v3:车辆/占道锚定主导 band 选择 —— 已因 Step 0 转向

> 出自 wb。依据 `docs/handoff/2026-07-16-cc-direction-v2-vehicle-anchor.md`(CC 方向 brief v2)。
> **状态: 阻塞1 (Step 0 前置检查) 已执行且失败 → v3 选择器主体暂不写码, 方案转向"候选缺失修复", 待 Jacob 决策。**

## 0. Step 0 前置检查结果 (阻塞1 —— 已执行, 失败)

脚本 `scripts/diag_candidate_pool_11.py`(只读, 复现 `_cv_v11:61-151` 候选生成, 不改生产代码)。
对 11 绿灯窗 ts∈[20.0,28.3] 62 个采样帧, 逐帧 dump 候选池, 统计与真斑马线 Y 区间 [373,463] 重叠的候选存在与否。

**硬数据:**
- 候选池含真带(373–463)候选的帧: **23/62 (37.1%)**
- 候选池无真带候选的帧: **39/62 (62.9%)**

**复现正确性核验:** ts=22.8(frame 684, 生产 CSV 该帧真带被选中 overlap=0.422) → 脚本 `has_true=1, true_cand_y=376-390` ✓; 无真带候选的 39 帧, argmax 顶部候选全部落在 **256–302 干扰带** ✓。

**判定:** 62.9% 的帧是**候选缺失**(纯条纹 `best_run` 在 band y0=344 未产出真带候选), 仅 37.1% 是"选择输了"(池里有真带但 argmax 输给干扰带)。
→ v3 的 occ 池化选择器**只能救那 37.1%, 对 62.9% 候选缺失帧完全无效**。写码无法解决 11 漏检。

**结论: 阻塞1 失败 → 按 cc 指令停, 不自行用车辆造带(违 C1), 升 Jacob 做方向决策。v3 选择器主体暂不写码。**

## 1. v2 作废声明
v2 时间稳定器参数全部作废。v3 不引入跨帧时间状态。

## 2. 阻塞2 改定 (cc 倾向 (i), 已采纳)
旧 `_cv_v11:140-147` 的 ×1.5–3.5 车辆中心加分**删除**。`grad_score = 行 137 纯条纹纹理 score`(车辆影响只走新 occ 池化, 不双重计数, 契合 C1)。

**措辞纠正:** 原 §3/§7 写的"回退与当前行为**逐字节一致**"是**错的** —— 删掉中心加分后, grad_score 已不同(当前生产用的是含 ×1.5–3.5 的版本, 且用的是全部 dets 框而非静止车框)。
→ 改为:**回退 = 纯梯度 argmax; 7 好视频不回退以全量 eval 逐视频确认, 不声称逐字节一致。** 某些好视频可能本就靠旧中心加分选对带, 删了要靠 eval 兜住。

## 3. 算法(选择器设计, 已完成但因 §0 暂不写码)
```
候选生成(行 81–151): 删 140–147 中心加分; grad_score = 行 137 纯条纹 score
stationary_vboxes = plumb 的静止车框(§4)
if vehicle_anchor and stationary_vboxes:
    for c in candidates:
        c.occ = max( compute_overlap_ratio(vb, 全宽矩形(cy1..cy2),
                                            footprint=0.5, denom="box")
                     for vb in stationary_vboxes )
    max_occ = max(c.occ)
    if max_occ >= occ_select_thr:
        pool = [c for c in candidates
                if c.occ >= occ_select_thr
                and c.grad_score >= occ_pool_grad_floor * max_grad]
        best = argmax(pool, key=grad_score)
    else:
        best = argmax(candidates, key=grad_score)   # C4 回退(纯梯度 argmax)
else:
    best = argmax(candidates, key=grad_score)
```
满足 C1/C2/C3/C4(详见 v2/v3 前文)。**但因 §0, 此选择器对 11 只覆盖 37% 帧, 不足以修复漏检。**

## 4. C2 — DAG 静止态 plumb(已查证可行, 不变)
`n_crosswalk`(dag.py:79) 在 `n_track`(76) 后, `ctx["states"]` 已含 `st["stationary"]`(tracker.py:63)。改 `n_crosswalk` 传静止车框。

## 5. 配置
```yaml
crosswalk:
  method: auto
  cv_min_area: 3000
  vehicle_anchor: true
  occ_select_thr: 0.1
  occ_pool_grad_floor: 0.5
```

## 6. 生命周期(reset)
`CrosswalkDetector.reset()`, `cli.run()` 顶部调用; `scripts/diag_mask_04_11.py` 04→11 前补 `reset()`。

## 7. 测试(TDD)
`tests/unit/test_crosswalk_v11_anchor.py`: 含"竞争边距真带 ~60px、静止车压真带 → 选真带"用例(镜像 11 失败几何) + "drift=40px 合理平移跟随"边界。阻塞2 改定后: 用例断言 grad_score 为纯条纹(不含中心加分)。

## 8. 验证口径
TDD → 全单测不回退 → 中间标尺(`diag_mask_04_11.py` 重跑看 11 绿灯窗 overlap)→ 全量 9 视频 eval(禁 `--reuse`)。**但 §0 已预判中间标尺对 11 仅部分改善(37%), 不足以判 v3 成功。**

## 9. 风险 A 已发生 → 升级方向(待 Jacob 决策)
**风险 A(候选缺失)已在 Step 0 证实(62.9% 帧无真带候选)。** 这不是选择器能救的。升级路径(均不违 C1, 因为候选仍纯条纹):

- **方向 X (推荐先试): 放宽条纹候选灵敏度** —— 改 `_cv_v11:81-151` 的候选生成:
  - `MIN_RUN_FRAC` 0.06→更低(如 0.04), 让真带短条纹段也能成 run
  - 自适应阈值 `1.5*std_val`→`1.2*std_val`, 让较弱条纹过线
  - 亮度门槛 `avg_brightness<100`→`<80`, 阴天/逆光帧不被砍
  - 目标: 让真带在更多帧生成候选, 之后再上 v3 选择器才有意义
  - 风险: 可能引入新 FP(噪声带变候选) → 靠 `occ_pool_grad_floor` + eval 兜
  - **这是独立改动, 需 Jacob 拍板是否动候选生成段**

- **方向 Y: 换 YOLO 分割模型(终极)** —— 彻底不依赖条纹扫描, 但需训练数据(PTL/12554/自标)。

- **方向 Z: 接受 11 漏检** —— 11 仅 9 视频之一, 总体 F1 影响有限; 但 11 是最高杠杆漏检, 放弃它 = 放弃最大增益点。

## 10. 改动文件清单
- 已落: `scripts/diag_candidate_pool_11.py`(Step 0 只读诊断, 已跑)
- 待 Jacob 决策后: `src/redlight/models/crosswalk.py` / `dag.py` / `config.yaml` / `tests/unit/test_crosswalk_v11_anchor.py`

## 11. 交回 CC / 升 Jacob 的决策点
1. **阻塞1 结果**: Step 0 显示 62.9% 候选缺失 → v3 选择器写码无意义(只救 37%), 已停。
2. **阻塞2**: 已选 (i)(删中心加分 + 去"逐字节一致"措辞)。
3. **待 Jacob 决策**: 走方向 X(放宽条纹灵敏度, 再上选择器) / Y(分割模型) / Z(接受漏检)?
   wb 建议先试 **方向 X**(改动小、不动架构、与 C1 兼容), 见效最快。
