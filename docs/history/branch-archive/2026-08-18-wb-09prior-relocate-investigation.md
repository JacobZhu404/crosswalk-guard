# 09 prior 重定位调查(根治线,承 prior-misframe)

> 结论前置:**单纯重定位 prior 坐标无法让 09 达到 P=1.000**。真正的 P 杀手是
> `[85,106]` 未知窗的一段 21s 确认 FP,原 prior 与新 prior 都会产生,`#3` 的 T 门控
> 也无能为力(整段 ≥ T)。建议**保留 #3 + 原 prior**;若要根治,需禁用 09 的
> prior 扩展(代码改动,待 cc 授权)。

## 背景 / cc 锚
cc brief:让 09 真持续行人绿在生产下检成整段长 run → 09 在 T=6 合法回 confirmed
(真绿 55.9s ≫ 6),同时保住 P=1.000。验收锚:09 confirmed 命中[11-72] + 01 仍 0 FP
+ 06/07/11 零回退。

09 GT(events.csv,`light_state` 列):
- `[0,10]` red,无违章
- `[11,72]` green,`is_violation=1`,`violating_plates=京AC63971;京NNM526`(两车!)
- `[72,106.4]` unknown,无违章

## 调查过程(只读诊断 → 定位 → 实测)

### ① detect 标定路径给出错误位置
`scripts/identify_pedestrian_signal.py` / `diag_09_lamp_locate.py`(走 `detect()` 跟踪亮斑)
给出 (0.65, 0.3)。但**生产 `observe()` 走的是 prior 直采 `_sample_roi`(160px ROI
按像素比例投票)**,两条路径不一致。实测 (0.65,0.3) 直采路径 **75% 帧返绿**(真绿窗仅占
57%),属于**车辆绿灯**(语义陷阱,正是 cc 警告的)。

### ② 在生产直采路径上 2D 栅格扫描(校正 GT 列名 `light_state`)
扫描发现真·行人灯位置 ≈ **(0.50, 0.10)**:
- `[11-72]` 绿窗 91% 帧判绿、`[0-10]` 红窗仅 11% 判绿、`[11-72]` 红判 0%
- 细网格:该处 `[11-72]` 有**连续 61s 完整绿 run**(整段真绿一次检出,完美)

### ③ 实测全管线(T=6,新 prior (0.50,0.10,40))
`verify_gates_01fp.py`(仅看 confirmed/review 计数与门逻辑):
- G1:违章01 `confirmed 1→0` ✓
- G2:06/07/09/11 零回退 ✓
- G4:09 min_max_run=102s,全 safe ✓
- ALL_PASS=True

但**该脚本不按 GT 窗校验每个违章区间**,掩盖了 FP。

### ④ 违章区间诊断(自建 `diag_09_violations.py`,按 GT 窗判定 P)
新 prior @ T=6:
```
CONFIRMED [12.66, 79.59]  中点窗=green(真违章)   ✓
CONFIRMED [85.12,106.26]  中点窗=unknown(无违章) ✗ FP
confirmed=2 review=0  → P=0.5
```

**回退到原 prior (0.2,0.35,160) @ T=6 对比:**
```
review   [ 1.48,  4.04]  中点窗=red(无违章)    ✓ #3 降级红窗瞬态假绿
CONFIRMED [12.66, 79.59]  中点窗=green(真违章) ✓
CONFIRMED [85.12,106.26]  中点窗=unknown(无违章) ✗ 同一 FP
confirmed=2 review=1  → P=0.5
```

## 根因
`[85-106]` FP 是**预先存在**的:该段 GT 为 unknown(灯灭/看不清),紧 ROI 变暗 →
prior 自适应扩展(`prior_roi_expand_factor=2.0`,160→320px / 40→80px)把邻近**车辆绿灯**
吃进来 → 假绿 → 叠加占道车辆 → 确认 FP。这是一条 21s 连续绿 run,`#3` 的 T 门控
(≤6s 降级)无法区分真绿 `[11-72]` 与假绿 `[72-106]`(二者被合并成一条 102s run)。

prior 重定位**不改变**这个 FP:新 prior 只是把红窗瞬态假绿并进了 102s run(丢了
review 标记,反而略差),对 `[85-106]` FP 毫无影响。

## 结论与建议
1. **单纯 prior 坐标重定位达不到 P=1.000** —— `[85-106]` FP 是真正的 P 杀手,与原
   prior 无关,`#3` 也压不住(21s ≥ T)。
2. **建议保留 #3 + 原 prior**:原 prior 下 #3 至少把红窗瞬态假绿标成 review(安全兜底),
   比新 prior(合并掉 review)更稳。
3. **若要根治 09 的 P**,需针对 `[72-106]` 灯灭期消除假绿:最小改动是让 09 的
   `prior_roi_expand_factor` 可逐 prior 配置(或对该视频禁用扩展),使灯灭期紧 ROI 读
   成 None 而非扩展到车辆绿。这是**代码改动**,超出"prior 重定位"范畴,需 cc 授权。
4. `diag_09_lamp_locate.py` 的 detect 标定路径会误匹配车辆绿灯,不可直接用于定 prior
   直采坐标 —— 已用 `diag_09_prior_grid.py`(生产直采路径)纠正。

## 验证脚本(本 worktree,未采纳 prior 改动)
- `scripts/diag_09_prior_sim.py` — 复刻 `_sample_roi`+扩展,扫 ROI 尺寸
- `scripts/diag_09_prior_grid.py` — 2D 栅格找真·行人灯位置(生产直采路径)
- `scripts/diag_09_prior_refine.py` — 按各 GT 窗连续绿 run 精修
- `scripts/diag_09_violations.py` — 提取 09 违章区间并按 GT 窗判 P(发现 FP 的关键)
